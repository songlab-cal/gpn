#!/usr/bin/env python3
"""
Standalone script for running GPN-star logits inference.

This script runs model inference to get logits at all positions in a parquet file.
It automatically detects GPUs and supports checkpointing for resumable inference.

Usage:
    python run_logits_inference.py <positions.parquet> <output.parquet> [options]

Example:
    python run_logits_inference.py \\
        results/positions/chr1/hg38/512/positions.parquet \\
        output/chr1_logits.parquet

    # With custom paths:
    python run_logits_inference.py \\
        positions.parquet \\
        output.parquet \\
        --msa_dir /path/to/msa \\
        --model_path /path/to/model \\
        --window_size 512

For multi-GPU usage, run with torchrun:
    torchrun --nproc_per_node=4 run_logits_inference.py ...
"""

import argparse
import os
import subprocess
import sys

# Default paths - modify these to match your setup
DEFAULT_MSA_DIR = "/accounts/grad/czye/GPN/gpn/examples/star/tmp/multiz100way/100"  # should contain all.zarr
DEFAULT_MODEL_PATH = "/accounts/grad/czye/GPN/gpn/examples/star/tmp/gpn-star-hg38-v100-200m"
DEFAULT_WINDOW_SIZE = 128
DEFAULT_PHYLO_DIST_PATH = "/accounts/grad/czye/GPN/gpn/examples/star/tmp/gpn-star-hg38-v100-200m/phylo_dist"


def get_num_gpus():
    """Detect the number of available GPUs."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return len(cuda_visible.split(","))
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True
        )
        return len(result.stdout.strip().split("\n"))
    except:
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run GPN-star logits inference on a positions file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python run_logits_inference.py positions.parquet output.parquet

  # With checkpointing for long runs
  python run_logits_inference.py positions.parquet output.parquet \\
      --checkpoint_batch_size 500000

  # With custom model and MSA paths
  python run_logits_inference.py positions.parquet output.parquet \\
      --msa_dir /path/to/msa \\
      --model_path /path/to/model

  # For multi-GPU, use torchrun directly:
  torchrun --nproc_per_node=4 -m gpn.star.inference logits positions.parquet msa_dir 512 checkpoint_dir output.parquet \\
      --per_device_batch_size 8 --is_file --checkpoint_batch_size 1000000 --cleanup_checkpoints
        """
    )
    
    # Required arguments
    parser.add_argument(
        "positions_file",
        help="Path to parquet file with 'chrom' and 'pos' columns"
    )
    parser.add_argument(
        "output_file",
        help="Path for output parquet file"
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--msa_dir",
        default=DEFAULT_MSA_DIR,
        help=f"Path to MSA zarr directory (default: {DEFAULT_MSA_DIR})"
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model checkpoint directory (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Genomic window size (default: {DEFAULT_WINDOW_SIZE})"
    )
    parser.add_argument(
        "--phylo_dist_path",
        type=str,
        default=DEFAULT_PHYLO_DIST_PATH,
        help=f"Path to phylogenetic distance directory (default: {DEFAULT_PHYLO_DIST_PATH})"
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=16,
        help="Batch size per GPU (default: 16)"
    )
    parser.add_argument(
        "--checkpoint_batch_size",
        type=int,
        default=1000000,
        help="Number of samples per checkpoint for resumable inference (default: 1000000)"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=None,
        help="Number of dataloader workers (auto-calculated based on CPUs/GPUs if not set)"
    )
    parser.add_argument(
        "--no_checkpointing",
        action="store_true",
        help="Disable batch checkpointing (not recommended for long runs)"
    )
    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Don't cleanup checkpoint files after successful completion"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.positions_file):
        print(f"Error: Positions file not found: {args.positions_file}")
        sys.exit(1)
    
    if not os.path.isdir(args.msa_dir):
        print(f"Error: MSA directory not found: {args.msa_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.model_path):
        print(f"Error: Model directory not found: {args.model_path}")
        sys.exit(1)
    
    # Detect GPUs
    num_gpus = args.num_gpus or get_num_gpus()
    
    # Calculate dataloader workers
    if args.dataloader_num_workers is None:
        num_cpus = os.cpu_count() or 1
        dataloader_num_workers = max(1, num_cpus // num_gpus)
    else:
        dataloader_num_workers = args.dataloader_num_workers
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Print configuration
    print("=" * 50)
    print("GPN-star Logits Inference")
    print("=" * 50)
    print(f"Positions file:     {args.positions_file}")
    print(f"MSA directory:      {args.msa_dir}")
    print(f"Model path:         {args.model_path}")
    print(f"Window size:        {args.window_size}")
    print(f"Output file:        {args.output_file}")
    print(f"Number of GPUs:     {num_gpus}")
    print(f"Per-device batch:   {args.per_device_batch_size}")
    print(f"Dataloader workers: {dataloader_num_workers}")
    print(f"Phylo dist path:    {args.phylo_dist_path}")
    if not args.no_checkpointing:
        print(f"Checkpoint batch:   {args.checkpoint_batch_size}")
        print(f"Cleanup checkpoints: {not args.keep_checkpoints}")
    else:
        print("Checkpointing:      disabled")
    print("=" * 50)
    
    # Build command
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m", "gpn.star.inference",
        "logits",
        args.positions_file,
        args.msa_dir,
        str(args.window_size),
        args.model_path,
        args.output_file,
        "--per_device_batch_size", str(args.per_device_batch_size),
        "--is_file",
        "--dataloader_num_workers", str(dataloader_num_workers),
    ]
    
    cmd.extend(["--phylo_dist_path", args.phylo_dist_path])
    
    if not args.no_checkpointing:
        cmd.extend(["--checkpoint_batch_size", str(args.checkpoint_batch_size)])
        if not args.keep_checkpoints:
            cmd.append("--cleanup_checkpoints")
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Set OMP_NUM_THREADS
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(dataloader_num_workers)
    
    # Run the command
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print("\n" + "=" * 50)
        print(f"Done! Output saved to: {args.output_file}")
        print("=" * 50)
    else:
        print(f"\nError: Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
