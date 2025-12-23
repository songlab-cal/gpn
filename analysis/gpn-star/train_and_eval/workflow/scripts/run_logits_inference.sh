#!/bin/bash
#
# Standalone script for running GPN-star logits inference
# 
# This script runs model inference to get logits at all positions in a parquet file.
# It automatically detects GPUs and supports checkpointing for resumable inference.
#
# Usage:
#   ./run_logits_inference.sh <positions.parquet> <output.parquet> [options]
#
# Example:
#   ./run_logits_inference.sh \
#       results/positions/chr1/hg38/512/positions.parquet \
#       output/chr1_logits.parquet
#
#   # With custom paths:
#   ./run_logits_inference.sh \
#       positions.parquet \
#       output.parquet \
#       --msa_dir /path/to/msa \
#       --model_path /path/to/model
#

set -e

# Default paths - modify these to match your setup
MSA_DIR="/accounts/grad/czye/GPN/gpn/examples/star/tmp/multiz100way/100" # should contain all.zarr
MODEL_PATH="/accounts/grad/czye/GPN/gpn/examples/star/tmp/gpn-star-hg38-v100-200m"
WINDOW_SIZE=128
PHYLO_DIST_PATH="/accounts/grad/czye/GPN/gpn/examples/star/tmp/gpn-star-hg38-v100-200m/phylo_dist"

# Check minimum arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <positions.parquet> <output.parquet> [options]"
    echo ""
    echo "Required arguments:"
    echo "  positions.parquet   Path to parquet file with 'chrom' and 'pos' columns"
    echo "  output.parquet      Path for output parquet file"
    echo ""
    echo "Optional arguments (pass after required args):"
    echo "  --msa_dir PATH              Path to MSA zarr directory (default: $MSA_DIR)"
    echo "  --model_path PATH           Path to model checkpoint directory (default: $MODEL_PATH)"
    echo "  --window_size N             Genomic window size (default: $WINDOW_SIZE)"
    echo "  --phylo_dist_path PATH      Path to phylogenetic distance directory (default: $PHYLO_DIST_PATH)"
    echo "  --per_device_batch_size N   Batch size per GPU (default: 16)"
    echo "  --checkpoint_batch_size N   Samples per checkpoint for resume (default: 1000000)"
    echo "  --dataloader_num_workers N  Number of dataloader workers (auto-calculated if not set)"
    echo "  --no_checkpointing          Disable batch checkpointing"
    echo "  --keep_checkpoints          Don't cleanup checkpoint files after completion"
    echo ""
    echo "Example:"
    echo "  $0 positions.parquet output.parquet --checkpoint_batch_size 500000"
    exit 1
fi

# Parse required arguments
POSITIONS_FILE="$1"
OUTPUT_FILE="$2"
shift 2

# Default values
PER_DEVICE_BATCH_SIZE=16
CHECKPOINT_BATCH_SIZE=1000000
DATALOADER_NUM_WORKERS=""
USE_CHECKPOINTING=true
CLEANUP_CHECKPOINTS=true

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --msa_dir)
            MSA_DIR="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --phylo_dist_path)
            PHYLO_DIST_PATH="$2"
            shift 2
            ;;
        --per_device_batch_size)
            PER_DEVICE_BATCH_SIZE="$2"
            shift 2
            ;;
        --checkpoint_batch_size)
            CHECKPOINT_BATCH_SIZE="$2"
            shift 2
            ;;
        --dataloader_num_workers)
            DATALOADER_NUM_WORKERS="$2"
            shift 2
            ;;
        --no_checkpointing)
            USE_CHECKPOINTING=false
            shift
            ;;
        --keep_checkpoints)
            CLEANUP_CHECKPOINTS=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ! -f "$POSITIONS_FILE" ]; then
    echo "Error: Positions file not found: $POSITIONS_FILE"
    exit 1
fi

if [ ! -d "$MSA_DIR" ]; then
    echo "Error: MSA directory not found: $MSA_DIR"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

# Detect number of GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        NUM_GPUS=1
    fi
fi

# Calculate dataloader workers if not specified
if [ -z "$DATALOADER_NUM_WORKERS" ]; then
    NUM_CPUS=$(nproc)
    DATALOADER_NUM_WORKERS=$((NUM_CPUS / NUM_GPUS))
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ -n "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "." ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "=============================================="
echo "GPN-star Logits Inference"
echo "=============================================="
echo "Positions file:     $POSITIONS_FILE"
echo "MSA directory:      $MSA_DIR"
echo "Model path:         $MODEL_PATH"
echo "Window size:        $WINDOW_SIZE"
echo "Output file:        $OUTPUT_FILE"
echo "Number of GPUs:     $NUM_GPUS"
echo "Per-device batch:   $PER_DEVICE_BATCH_SIZE"
echo "Dataloader workers: $DATALOADER_NUM_WORKERS"
if [ -n "$PHYLO_DIST_PATH" ]; then
    echo "Phylo dist path:    $PHYLO_DIST_PATH"
fi
if [ "$USE_CHECKPOINTING" = true ]; then
    echo "Checkpoint batch:   $CHECKPOINT_BATCH_SIZE"
    echo "Cleanup checkpoints: $CLEANUP_CHECKPOINTS"
else
    echo "Checkpointing:      disabled"
fi
echo "=============================================="

export OMP_NUM_THREADS=$DATALOADER_NUM_WORKERS

# Build the command
CMD="torchrun --nproc_per_node=$NUM_GPUS -m gpn.star.inference logits"
CMD="$CMD $POSITIONS_FILE $MSA_DIR $WINDOW_SIZE $MODEL_PATH $OUTPUT_FILE"
CMD="$CMD --per_device_batch_size $PER_DEVICE_BATCH_SIZE"
CMD="$CMD --is_file"
CMD="$CMD --dataloader_num_workers $DATALOADER_NUM_WORKERS"

if [ -n "$PHYLO_DIST_PATH" ]; then
    CMD="$CMD --phylo_dist_path $PHYLO_DIST_PATH"
fi

if [ "$USE_CHECKPOINTING" = true ]; then
    CMD="$CMD --checkpoint_batch_size $CHECKPOINT_BATCH_SIZE"
    if [ "$CLEANUP_CHECKPOINTS" = true ]; then
        CMD="$CMD --cleanup_checkpoints"
    fi
fi

echo "Running: $CMD"
echo ""

# Run the inference
eval $CMD

echo ""
echo "=============================================="
echo "Done! Output saved to: $OUTPUT_FILE"
echo "=============================================="
