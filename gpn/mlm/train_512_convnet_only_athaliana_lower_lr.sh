WANDB_PROJECT=PlantBERT_MLM_512 python ./run_mlm_custom.py \
    --report_to wandb \
    --run_name ConvNet_only_athaliana_lower_lr_v2 \
    --do_train \
    --do_eval \
    --train_fasta_path ../../data/mlm/dataset/train/Arabidopsis_thaliana.train.parquet \
    --validation_file ../../data/mlm/dataset/test/Arabidopsis_thaliana.test.512.256.parquet \
    --model_type ConvNet \
    --line_by_line True \
    --window_size 512 \
    --learning_rate 1e-4 \
    --save_strategy steps \
    --save_steps 20000 \
    --max_steps 200000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --warmup_steps 10000 \
    --logging_steps 10000 \
    --output_dir results_512_convnet_only_athaliana_lower_lr_v2 \
    --tokenizer_name ../../data/mlm/tokenizer_bare \
    --per_device_train_batch_size 250 \
    --per_device_eval_batch_size 250 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --adam_epsilon 1e-4 \
    --seed 50 \
    --prediction_loss_only True \
    --lr_scheduler_type constant_with_warmup \
    --model_name_or_path ./results_512_convnet_only_athaliana/checkpoint-1000000/ \

#    --config_overrides vocab_size=6 \

#    --resume_from_checkpoint ./results_512_convnet_only_athaliana/checkpoint-1300000 \
#    --ignore_data_skip \