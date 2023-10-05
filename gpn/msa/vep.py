import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM

import gpn.model
from gpn.data import Tokenizer, ReverseComplementer


class MLMforVEPModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)

    def get_llr(self, input_ids, aux_features, pos, ref, alt):
        logits = self.model.forward(
            input_ids=input_ids, aux_features=aux_features
        ).logits
        logits = logits[torch.arange(len(pos)), pos]
        # TODO: maybe [:, ref] would work?
        logits_ref = logits[torch.arange(len(ref)), ref]
        logits_alt = logits[torch.arange(len(alt)), alt]
        llr = logits_alt - logits_ref
        return llr

    def forward(
        self,
        input_ids_fwd=None,
        aux_features_fwd=None,
        pos_fwd=None,
        ref_fwd=None,
        alt_fwd=None,
        input_ids_rev=None,
        aux_features_rev=None,
        pos_rev=None,
        ref_rev=None,
        alt_rev=None,
    ):
        llr_fwd = self.get_llr(
            input_ids_fwd, aux_features_fwd, pos_fwd, ref_fwd, alt_fwd
        )
        llr_rev = self.get_llr(
            input_ids_rev, aux_features_rev, pos_rev, ref_rev, alt_rev
        )
        llr = (llr_fwd + llr_rev) / 2
        return llr


class VEPInference(object):
    def __init__(self, model_path, genome_msa, window_size, disable_aux_features=False):
        self.model = MLMforVEPModel(model_path)
        self.genome_msa = genome_msa
        self.window_size = window_size
        self.disable_aux_features = disable_aux_features
        self.reverse_complementer = ReverseComplementer()
        self.tokenizer = Tokenizer()

    def tokenize_function(self, V):
        # we convert from 1-based coordinate (standard in VCF) to
        # 0-based, to use with GenomeMSA
        chrom = np.array(V["chrom"])
        pos = np.array(V["pos"]) - 1
        start = pos - self.window_size // 2
        end = pos + self.window_size // 2
        msa_fwd, msa_rev = self.genome_msa.get_msa_batch_fwd_rev(
            chrom,
            start,
            end,
            tokenize=True,
        )
        pos_fwd = self.window_size // 2
        pos_rev = pos_fwd - 1 if self.window_size % 2 == 0 else pos_fwd

        ref_fwd = np.array(
            [np.frombuffer(x.encode("ascii"), dtype="S1") for x in V["ref"]]
        )
        alt_fwd = np.array(
            [np.frombuffer(x.encode("ascii"), dtype="S1") for x in V["alt"]]
        )
        ref_rev = self.reverse_complementer(ref_fwd)
        alt_rev = self.reverse_complementer(alt_fwd)

        def prepare_output(msa, pos, ref, alt):
            ref, alt = self.tokenizer(ref.flatten()), self.tokenizer(alt.flatten())
            input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]
            assert (
                input_ids[:, pos] == ref
            ).all(), f"{input_ids[:, pos].tolist()}, {ref.tolist()}"
            input_ids[:, pos] = self.tokenizer.mask_token_id()
            input_ids = input_ids.astype(np.int64)
            pos = np.full(len(input_ids), pos)
            ref, alt = ref.astype(np.int64), alt.astype(np.int64)
            return input_ids, aux_features, pos, ref, alt

        res = {}
        (
            res["input_ids_fwd"],
            res["aux_features_fwd"],
            res["pos_fwd"],
            res["ref_fwd"],
            res["alt_fwd"],
        ) = prepare_output(msa_fwd, pos_fwd, ref_fwd, alt_fwd)
        (
            res["input_ids_rev"],
            res["aux_features_rev"],
            res["pos_rev"],
            res["ref_rev"],
            res["alt_rev"],
        ) = prepare_output(msa_rev, pos_rev, ref_rev, alt_rev)
        if self.disable_aux_features:
            del res["aux_features_fwd"]
            del res["aux_features_rev"]
        return res

    def postprocess(self, pred):
        return pd.DataFrame(pred, columns=["score"])
