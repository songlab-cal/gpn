import argparse
import tempfile

import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

from gpn import register_auto_classes
from gpn.data import Genome, load_dataset_from_file_or_dir, token_input_id
from gpn.scoring import require_reference_matches, validate_snv_batch
from gpn.star.checkpoint import write_dataframe_atomic


class MLMforVEPModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        register_auto_classes("ss")
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()

    def get_llr(self, input_ids, pos, ref, alt):
        logits = self.model.forward(input_ids=input_ids).logits
        logits = logits[torch.arange(len(pos)), pos]
        logits_ref = logits[torch.arange(len(ref)), ref]
        logits_alt = logits[torch.arange(len(alt)), alt]
        llr = logits_alt - logits_ref
        return llr

    def forward(
        self,
        input_ids_fwd=None,
        pos_fwd=None,
        ref_fwd=None,
        alt_fwd=None,
        input_ids_rev=None,
        pos_rev=None,
        ref_rev=None,
        alt_rev=None,
    ):
        llr_fwd = self.get_llr(input_ids_fwd, pos_fwd, ref_fwd, alt_fwd)
        llr_rev = self.get_llr(input_ids_rev, pos_rev, ref_rev, alt_rev)
        llr = (llr_fwd + llr_rev) / 2
        return llr


def _tokenize_variant_batch(vs, genome, window_size, tokenizer, n_prefix=0):
    chromosomes, positions, references, alternates = validate_snv_batch(
        vs["chrom"], vs["pos"], vs["ref"], vs["alt"]
    )
    chrom = np.array(chromosomes)
    n = len(chrom)
    pos = np.array(positions) - 1
    start = pos - window_size // 2
    end = pos + window_size // 2
    if window_size % 2 == 1:
        end += 1
    seq_fwd, seq_rev = zip(
        *(genome.get_seq_fwd_rev(chrom[i], start[i], end[i]) for i in range(n))
    )
    seq_fwd = np.array([list(seq.upper()) for seq in seq_fwd], dtype="object")
    seq_rev = np.array([list(seq.upper()) for seq in seq_rev], dtype="object")
    if seq_fwd.ndim != 2 or seq_fwd.shape[1] != window_size:
        raise ValueError("Forward genome windows do not match window_size")
    if seq_rev.ndim != 2 or seq_rev.shape[1] != window_size:
        raise ValueError("Reverse genome windows do not match window_size")
    ref_fwd = np.array(references)
    alt_fwd = np.array(alternates)
    ref_rev = np.array([str(Seq(x).reverse_complement()) for x in ref_fwd])
    alt_rev = np.array([str(Seq(x).reverse_complement()) for x in alt_fwd])
    pos_fwd = window_size // 2
    pos_rev = pos_fwd - 1 if window_size % 2 == 0 else pos_fwd

    def tokenize(seqs):
        return tokenizer(
            seqs,
            padding=False,
            truncation=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=False,
        )["input_ids"]

    def prepare_output(seq, center, ref, alt, orientation):
        require_reference_matches(
            seq[:, center],
            ref,
            chromosomes,
            positions,
            orientation=orientation,
        )
        seq[:, center] = tokenizer.mask_token
        return (
            tokenize(["".join(x) for x in seq]),
            [center + n_prefix for _ in range(n)],
            [token_input_id(x, tokenizer, n_prefix) for x in ref],
            [token_input_id(x, tokenizer, n_prefix) for x in alt],
        )

    res = {}
    (
        res["input_ids_fwd"],
        res["pos_fwd"],
        res["ref_fwd"],
        res["alt_fwd"],
    ) = prepare_output(seq_fwd, pos_fwd, ref_fwd, alt_fwd, "forward")
    (
        res["input_ids_rev"],
        res["pos_rev"],
        res["ref_rev"],
        res["alt_rev"],
    ) = prepare_output(seq_rev, pos_rev, ref_rev, alt_rev, "reverse-complement")
    return res


def run_vep(
    variants,
    genome,
    window_size,
    tokenizer,
    model,
    n_prefix=0,
    per_device_batch_size=8,
    dataloader_num_workers=0,
    fp16=None,
    torch_compile=None,
):
    def get_tokenized_seq(vs):
        return _tokenize_variant_batch(
            vs,
            genome,
            window_size,
            tokenizer,
            n_prefix=n_prefix,
        )

    variants.set_transform(get_tokenized_seq)
    temporary_output_dir = tempfile.TemporaryDirectory(prefix="gpn-vep-")
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=temporary_output_dir.name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        torch_compile=use_cuda if torch_compile is None else torch_compile,
        fp16=use_cuda if fp16 is None else fp16,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args)
    trainer._gpn_temporary_output_dir = temporary_output_dir
    predictions = trainer.predict(test_dataset=variants).predictions
    if not trainer.accelerator.is_main_process:
        return None
    return predictions


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run GPN zero-shot variant effect prediction"
    )
    parser.add_argument(
        "variants_path",
        type=str,
        help="Variants path. Needs the following columns: chrom,pos,ref,alt. pos should be 1-based",
    )
    parser.add_argument(
        "genome_path",
        type=str,
        help="Genome path (fasta, potentially compressed)",
    )
    parser.add_argument("window_size", type=int, help="Genomic window size")
    parser.add_argument("model_path", help="Model path (local or on HF hub)", type=str)
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--per-device-batch-size",
        "--per_device_batch_size",
        dest="per_device_batch_size",
        help="Per device batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tokenizer-path",
        "--tokenizer_path",
        dest="tokenizer_path",
        type=str,
        help="Tokenizer path (optional, else will use model_path)",
    )
    parser.add_argument(
        "--n-prefix",
        "--n_prefix",
        dest="n_prefix",
        type=int,
        default=0,
        help="Number of prefix tokens (e.g. CLS).",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        "--dataloader_num_workers",
        dest="dataloader_num_workers",
        type=int,
        default=0,
        help="Dataloader num workers",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--is-file",
        "--is_file",
        dest="is_file",
        action="store_true",
        help="VARIANTS_PATH is a file, not directory",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16 inference (default: enabled when CUDA is available)",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use torch.compile (default: enabled when CUDA is available)",
    )
    return parser


def _validate_cli_args(parser, args):
    if args.window_size <= 0:
        parser.error("window_size must be positive")
    if args.per_device_batch_size <= 0:
        parser.error("--per-device-batch-size must be positive")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader-num-workers must be non-negative")
    if args.n_prefix < 0:
        parser.error("--n-prefix must be non-negative")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)

    variants = load_dataset_from_file_or_dir(
        args.variants_path,
        split=args.split,
        is_file=args.is_file,
    )
    genome = Genome(args.genome_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_path
    )
    model = MLMforVEPModel(args.model_path)
    pred = run_vep(
        variants,
        genome,
        args.window_size,
        tokenizer,
        model,
        per_device_batch_size=args.per_device_batch_size,
        n_prefix=args.n_prefix,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        torch_compile=args.torch_compile,
    )
    if pred is not None:
        write_dataframe_atomic(pd.DataFrame(pred, columns=["score"]), args.output_path)


if __name__ == "__main__":
    main()
