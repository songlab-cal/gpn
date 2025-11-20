dataset_path=$1
output_path=$2
per_device_batch_size=$3
n_gpu=$4

total_batch_size=2048
gradient_accumulation_steps=$((total_batch_size / (per_device_batch_size * n_gpu)))

num_hidden_layers=64
hidden_size=1024

version=first_steps
name=${version}

if [ "$n_gpu" -eq 1 ]; then
    python_cmd="python"
else
    python_cmd="torchrun --nnodes 1 --nproc_per_node $n_gpu"
fi

OMP_NUM_THREADS=8 \
WANDB_PROJECT=gpn-animal-promoter \
    ${python_cmd} \
    -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name ${dataset_path} \
    --soft_masked_loss_weight_train 0.01 \
    --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 8 \
    --seed 42 \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 1000 \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size ${per_device_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --total_batch_size ${total_batch_size} \
    --max_steps 10000 \
    --warmup_steps 1000 \
    --run_name ${name} \
    --output_dir ${output_path} \
    --torch_compile \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine \
    --ddp_find_unused_parameters False \
    --bf16 \
    --bf16_full_eval \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --model_type GPN --config_overrides hidden_size=${hidden_size},num_hidden_layers=${num_hidden_layers},embedding=embedding,encoder=bytenet,mlm_head_transform=False,slim=True,bias=False
