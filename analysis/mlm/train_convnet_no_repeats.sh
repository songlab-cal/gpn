WANDB_PROJECT=GPN_Arabidopsis_no_repeats_4 python ./run_mlm_custom.py \
    --report_to wandb \
    --run_name ConvNet_M500k_30L_1e-3_accum4 \
    --do_train \
    --do_eval \
    --train_fasta_path ../../data/mlm/dataset/train/Arabidopsis_thaliana.train.parquet \
    --validation_file ../../data/mlm/dataset/test/Arabidopsis_thaliana.test.512.256.parquet \
    --model_type ConvNet \
    --config_overrides n_layers=30 \
    --line_by_line True \
    --window_size 512 \
    --learning_rate 1e-3 \
    --save_strategy steps \
    --save_steps 10000 \
    --max_steps 500000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --warmup_steps 10000 \
    --logging_steps 10000 \
    --output_dir results/checkpoints/nr4_convnet_m500k_30l_1e-3_accum4 \
    --tokenizer_name ../../data/mlm/tokenizer_bare2 \
    --per_device_train_batch_size 250 \
    --per_device_eval_batch_size 250 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --adam_epsilon 1e-4 \
    --seed 42 \
    --prediction_loss_only True \
    --lr_scheduler_type cosine \
    --remove_unused_columns False \

#--resume_from_checkpoint ./results/checkpoints/convnet_v3/checkpoint-800000 \
#--ignore_data_skip \