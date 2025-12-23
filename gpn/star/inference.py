import argparse
from datasets import load_dataset, disable_caching, Dataset
import os
import tempfile
from transformers import Trainer, TrainingArguments
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
import torch.distributed as dist

from gpn.star.data import load_dataset_from_file_or_dir
from gpn.star.data import GenomeMSA
from gpn.star.vep import VEPInference
from gpn.star.logits import LogitsInference
from gpn.star.embedding import EmbeddingInference
from gpn.star.vep_embedding import VEPEmbeddingInference
from gpn.star.utils import find_directory_sum_paths

disable_caching()


def is_main_process():
    """Check if this is the main process (rank 0) in distributed training."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size():
    """Get the number of processes in distributed training."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    """Synchronization barrier for distributed training."""
    if dist.is_initialized():
        dist.barrier()


class_mapping = {
    "vep": VEPInference,
    "logits": LogitsInference,
    "embedding": EmbeddingInference,
    "vep_embedding": VEPEmbeddingInference,
}


def run_inference(
    dataset,
    inference,
    per_device_batch_size=8,
    dataloader_num_workers=0,
):
    dataset.set_transform(inference.tokenize_function)
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        torch_compile=True,
        fp16=True,
        report_to="none",  # Disable wandb/tensorboard logging prompts
    )
    print(dataset)
    trainer = Trainer(model=inference.model, args=training_args)
    pred = trainer.predict(test_dataset=dataset).predictions
    return inference.postprocess(pred)


def run_inference_with_checkpoints(
    dataset,
    inference,
    checkpoint_dir,
    checkpoint_batch_size,
    per_device_batch_size=8,
    dataloader_num_workers=0,
):
    """
    Run inference with batch checkpointing and resume capability.
    
    Works correctly with distributed training (torchrun with multiple GPUs).
    
    Args:
        dataset: The full dataset to process
        inference: The inference object
        checkpoint_dir: Directory to save intermediate batch results
        checkpoint_batch_size: Number of samples per checkpoint batch
        per_device_batch_size: Batch size per device for inference
        dataloader_num_workers: Number of dataloader workers
    
    Returns:
        DataFrame with predictions for the entire dataset
    """
    main_process = is_main_process()
    world_size = get_world_size()
    
    # Only main process creates the checkpoint directory
    if main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
    barrier()  # Wait for directory to be created
    
    total_samples = len(dataset)
    
    # Adjust checkpoint_batch_size to be divisible by world_size for even distribution
    # This ensures each GPU gets the same number of samples per batch
    if world_size > 1:
        # Round up to nearest multiple of world_size
        checkpoint_batch_size = ((checkpoint_batch_size + world_size - 1) // world_size) * world_size
    
    num_batches = (total_samples + checkpoint_batch_size - 1) // checkpoint_batch_size
    
    if main_process:
        print(f"Total samples: {total_samples}")
        print(f"Checkpoint batch size: {checkpoint_batch_size}")
        print(f"Number of batches: {num_batches}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"World size (num GPUs): {world_size}")
    
    # Find already completed batches (all processes need to know this)
    completed_batches = set()
    for f in glob(os.path.join(checkpoint_dir, "batch_*.parquet")):
        batch_idx = int(os.path.basename(f).replace("batch_", "").replace(".parquet", ""))
        completed_batches.add(batch_idx)
    
    if completed_batches and main_process:
        print(f"Found {len(completed_batches)} completed batches, resuming...")
    
    # Process each batch
    batch_iterator = tqdm(range(num_batches), desc="Processing batches") if main_process else range(num_batches)
    for batch_idx in batch_iterator:
        batch_file = os.path.join(checkpoint_dir, f"batch_{batch_idx:06d}.parquet")
        
        if batch_idx in completed_batches:
            if main_process:
                print(f"Batch {batch_idx} already completed, skipping...")
            continue
        
        start_idx = batch_idx * checkpoint_batch_size
        end_idx = min(start_idx + checkpoint_batch_size, total_samples)
        
        if main_process:
            print(f"Processing batch {batch_idx}: samples {start_idx} to {end_idx}")
        
        # Select subset of dataset - all processes must select the same range
        batch_dataset = dataset.select(range(start_idx, end_idx))
        
        # Run inference on this batch
        # The Trainer will handle sharding across GPUs internally
        batch_pred = run_inference(
            batch_dataset,
            inference,
            per_device_batch_size=per_device_batch_size,
            dataloader_num_workers=dataloader_num_workers,
        )
        
        # Only main process saves the checkpoint (predictions are already gathered)
        if main_process:
            batch_pred.to_parquet(batch_file, index=False)
            print(f"Saved batch {batch_idx} to {batch_file}")
        
        # Synchronize all processes before moving to next batch
        barrier()
    
    # Combine all batches (only on main process, but all need to return)
    if main_process:
        print("Combining all batches...")
        all_batch_files = sorted(glob(os.path.join(checkpoint_dir, "batch_*.parquet")))
        all_preds = pd.concat([pd.read_parquet(f) for f in all_batch_files], ignore_index=True)
    else:
        # Non-main processes return empty DataFrame (won't be used)
        all_preds = pd.DataFrame()
    
    barrier()  # Ensure all processes finish together
    return all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with AutoModelForMaskedLM",
    )
    parser.add_argument(
        "command",
        type=str,
        help="""Command to run:
        - vep: zero-shot variant effect prediction (LLR)
        - logits: masked language model logits
        - embedding: averaged embedding from last layer
        """,
        choices=["vep", "logits", "embedding", "vep_embedding"],
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="""Input path, either HF dataset, parquet, csv/tsv, vcf, with columns:
        - vep: chrom, pos, ref, alt
        - logits: chrom, pos
        - embedding: chrom, start, end
        """,
    )
    parser.add_argument(
        "msa_path",
        type=str,
        help="Genome MSA path (zarr)",
    )
    parser.add_argument("window_size", type=int, help="Genomic window size")    
    parser.add_argument("model_path", help="Model path (local or on HF hub)", type=str)
    parser.add_argument("output_path", help="Output path (parquet)", type=str)
    parser.add_argument(
        "--per_device_batch_size",
        help="Per device batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0, help="Dataloader num workers"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--is_file",
        action="store_true",
        help="VARIANTS_PATH is a file, not directory",
    )
    parser.add_argument(
        "--disable_aux_features",
        action="store_true",
    )
    parser.add_argument(
        "--center_window_size",
        type=int,
        help="[embedding] Genomic window size to average at the center of the windows",
    )
    parser.add_argument(
        "--checkpoint_batch_size",
        type=int,
        default=None,
        help="Number of samples per checkpoint batch. If set, enables batch checkpointing "
             "with resume capability. Intermediate results are saved to a checkpoint directory "
             "next to the output file. Useful for long-running jobs that may be interrupted.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Custom checkpoint directory. If not set, defaults to output_path + '_checkpoints'",
    )
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Remove checkpoint directory after successfully combining all batches",
    )
    parser.add_argument(
        "--phylo_dist_path",
        type=str,
        default=None,
        help="Path to phylogenetic distance directory. If not set, defaults to the stored one in the checkpoint",
    )
    args = parser.parse_args()
    print(args)

    try:
        dataset = load_dataset_from_file_or_dir(
            args.input_path,
            split=args.split,
            is_file=args.is_file,
        )
    except:
        dataset = Dataset.from_pandas(pd.read_parquet(args.input_path+'/test.parquet'))

    msa_paths = find_directory_sum_paths(args.msa_path)
    genome_msa_list = [GenomeMSA(path, n_species=n_species, subset_chroms=dataset.unique("chrom"), in_memory=False) 
                       for n_species, path in msa_paths.items()]
    
    # sorry this is hacky, should use subparsers
    kwargs = {}
    if args.command == "embedding":
        kwargs["center_window_size"] = args.center_window_size
    if args.phylo_dist_path is not None:
        kwargs["phylo_dist_path"] = args.phylo_dist_path
    
    inference = class_mapping[args.command](
        args.model_path,
        genome_msa_list,
        args.window_size,
        disable_aux_features=args.disable_aux_features,
        **kwargs,
    )
    
    # Use checkpointing if checkpoint_batch_size is specified
    if args.checkpoint_batch_size is not None:
        checkpoint_dir = args.checkpoint_dir or (args.output_path + "_checkpoints")
        pred = run_inference_with_checkpoints(
            dataset,
            inference,
            checkpoint_dir=checkpoint_dir,
            checkpoint_batch_size=args.checkpoint_batch_size,
            per_device_batch_size=args.per_device_batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
        )
        # Only main process handles file operations
        if is_main_process():
            # Optionally cleanup checkpoints after successful completion
            if args.cleanup_checkpoints:
                import shutil
                shutil.rmtree(checkpoint_dir)
                print(f"Cleaned up checkpoint directory: {checkpoint_dir}")
            
            directory = os.path.dirname(args.output_path)
            if directory != "" and not os.path.exists(directory):
                os.makedirs(directory)
            pred.to_parquet(args.output_path, index=False)
    else:
        pred = run_inference(
            dataset,
            inference,
            per_device_batch_size=args.per_device_batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
        )
        directory = os.path.dirname(args.output_path)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)
        pred.to_parquet(args.output_path, index=False)
