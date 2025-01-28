#!/bin/bash
dataset=$1
num_hidden_layers=$2
hidden_size=$3
per_device_batch_size=$4
n_gpu=$5

total_batch_size=2048
gradient_accumulation_steps=$((total_batch_size / (per_device_batch_size * n_gpu)))

version=v4
name=${version}_${dataset}_${num_hidden_layers}_${hidden_size}

if [ "$n_gpu" -eq 1 ]; then
    python_cmd="python"
else
    python_cmd="torchrun --nnodes 1 --nproc_per_node $n_gpu"
fi

#SBATCH --job-name=${subset}
#SBATCH --output=output_%x.out
#SBATCH --error=error_%x.err
#SBATCH --partition=yss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --time=6:00:00

#mamba activate gpn  # didn't get this to work yet

OMP_NUM_THREADS=8 \
WANDB_PROJECT=gpn-animal-promoter \
    ${python_cmd} \
    -m gpn.ss.run_mlm --do_train --do_eval \
    --report_to wandb --prediction_loss_only True --remove_unused_columns False \
    --dataset_name results/dataset/${dataset} --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.01 --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 --optim adamw_torch \
    --dataloader_num_workers 8 --preprocessing_num_workers 8 --seed 42 \
    --save_strategy steps --save_steps 10000 --evaluation_strategy steps --eval_steps 10000 --logging_steps 10000 --max_steps 1000000 --warmup_steps 1000 \
    --load_best_model_at_end \
    --per_device_train_batch_size ${per_device_batch_size} --per_device_eval_batch_size ${per_device_batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} --total_batch_size ${total_batch_size} \
    --model_type GPN --config_overrides hidden_size=${hidden_size},num_hidden_layers=${num_hidden_layers},embedding=embedding,encoder=bytenet,mlm_head_transform=False,slim=True,bias=False \
    --run_name ${name} --output_dir results/checkpoints/mlm/${name} \
    --torch_compile \
    --learning_rate 1e-3 --lr_scheduler_type constant_with_warmup \
    --ddp_find_unused_parameters False \
    --bf16 --bf16_full_eval \

#    --learning_rate 1e-3 --lr_scheduler_type cosine_with_min_lr --min_lr_rate 0.1 \

#OMP_NUM_THREADS=8  torchrun --standalone --nnodes 1 --nproc_per_node 4 \
# --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 --total_batch_size 2048 \
#    --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --gradient_accumulation_steps 4 --total_batch_size 2048 \
#    --save_strategy steps --save_steps 10000 --evaluation_strategy steps --eval_steps 10000 --logging_steps 10000 --max_steps 1000000 --warmup_steps 1000 \