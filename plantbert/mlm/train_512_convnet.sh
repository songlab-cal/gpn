WANDB_PROJECT=PlantBERT_MLM_512_NO_REPEATS python ./run_mlm_custom.py \
    --report_to wandb \
    --run_name ConvNet \
    --do_train \
    --do_eval \
    --model_type ConvNet \
    --train_fasta_path ../../data/mlm/dataset/Ath.train.parquet \
    --validation_file ../../data/mlm/dataset/Ath.test.512.256.parquet \
    --line_by_line True \
    --window_size 512 \
    --learning_rate 1e-3 \
    --save_strategy steps \
    --save_steps 100000 \
    --max_steps 1000000 \
    --evaluation_strategy steps \
    --eval_steps 100000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --warmup_steps 10000 \
    --logging_steps 100000 \
    --output_dir results/checkpoints/512_no_repeats/convnet \
    --tokenizer_name ../../data/mlm/tokenizer_bare2 \
    --per_device_train_batch_size 250 \
    --per_device_eval_batch_size 250 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --adam_epsilon 1e-4 \
    --seed 42 \
    --prediction_loss_only True \
    --lr_scheduler_type constant_with_warmup \
    --remove_unused_columns False \


#    --resume_from_checkpoint ./results_512_convnet/checkpoint-800000 \
#    --ignore_data_skip \
