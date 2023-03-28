WANDB_PROJECT=GPN_Arabidopsis_multispecies python -m gpn.run_mlm --do_train --do_eval \
    --fp16 --report_to wandb --prediction_loss_only True --remove_unused_columns False \
    --dataset_name output/merged_dataset/balanced --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.0 --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 --optim adamw_torch \
    --dataloader_num_workers 16 --preprocessing_num_workers 32 --seed 42 \
    --save_strategy steps --save_steps 10000 --evaluation_strategy steps --eval_steps 10000 --logging_steps 10000 --max_steps 120000 --warmup_steps 1000 \
    --learning_rate 1e-3 --lr_scheduler_type constant_with_warmup \
    --run_name ConvNet_batch2048_weight0 --output_dir /scratch/users/gbenegas/checkpoints/GPN_Arabidopsis_multispecies/ConvNet_batch2048_weight0 --model_type ConvNet \
    --per_device_train_batch_size 2048 --per_device_eval_batch_size 2048 --gradient_accumulation_steps 1 \
    --resume_from_checkpoint /scratch/users/gbenegas/checkpoints/GPN_Arabidopsis_multispecies/ConvNet_batch2048_weight0/checkpoint-30000 \
    --ignore_data_skip \

# note: there's a bug in huggingface trainer with iterable dataset, the per_device_*_batch_size is interpreted as total batch size

#    --run_name RoFormer_12_weight0.5_v2 --output_dir /scratch/users/gbenegas/checkpoints/GPN_Arabidopsis_7/RoFormer_12_weight0.5_v2 --model_type roformer --config_overrides vocab_size=7 \

#     --torch_compile \  # not working, will wait until stable version
# --save_strategy steps --save_steps 10000 --evaluation_strategy steps --eval_steps 5000 \
#    --logging_steps 5000 --max_steps 50000 --warmup_steps 500 \
    #--save_strategy epoch --evaluation_strategy epoch --num_train_epochs 1 --warmup_ratio 0.01 \
    #--run_name RoFormer_8 --output_dir output/checkpoints/RoFormer_8 --model_type roformer --config_overrides vocab_size=7,num_hidden_layers=8,num_attention_heads=8,hidden_size=512,intermediate_size=2048 \

#    --learning_rate 1e-4 --lr_scheduler_type cosine \
#    --run_name RoFormer_12 --output_dir output/checkpoints/RoFormer_12 --model_type roformer --config_overrides vocab_size=7 \
#    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 2 \

#WANDB_PROJECT=GPN_Arabidopsis_6 python -m gpn.run_mlm --do_train --do_eval \
#    --fp16 --report_to wandb --prediction_loss_only True --remove_unused_columns False \
#    --dataset_name output/dataset/mlm/gpn --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
#    --soft_masked_loss_weight_train 1.0 --soft_masked_loss_weight_evaluation 1.0 \
#    --weight_decay 0.01 --optim adamw_torch \
#    --dataloader_num_workers 16 --preprocessing_num_workers 32 --seed 42 \
#    --save_strategy steps --save_steps 10000 --evaluation_strategy steps --eval_steps 10000 --logging_steps 10000 --max_steps 400000 --warmup_steps 1000 \
#    --learning_rate 1e-3 --lr_scheduler_type constant_with_warmup \
#    --run_name ConvNet_25_weight1.0_batch256 --output_dir /scratch/users/gbenegas/checkpoints/GPN_Arabidopsis_6/ConvNet_25_weight1.0_batch256 --model_type ConvNet \
#    --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 \