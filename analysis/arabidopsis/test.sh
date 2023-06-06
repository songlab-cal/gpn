WANDB_PROJECT=GPN_Arabidopsis_multispecies python -m gpn.run_mlm --do_test \
    --fp16 --prediction_loss_only True --remove_unused_columns False \
    --dataset_name output/merged_dataset/512/256/True/balanced --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_test 0.0 \
    --dataloader_num_workers 16 --preprocessing_num_workers 32 --seed 42 \
    --output_dir /tmp/dir \
    --per_device_eval_batch_size 512 \
    --model_name_or_path /scratch/users/gbenegas/checkpoints/GPN_Arabidopsis_multispecies/ConvNet_batch2048_secondpart/checkpoint-30000 \

# note: there's a bug in huggingface trainer with iterable dataset, the per_device_*_batch_size is interpreted as total batch size
