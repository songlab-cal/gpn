python ./run_mlm.py \
    --do_train \
    --train_fasta_path ./all.contigs.fa.gz \
    --do_eval \
    --validation_file ./val_seqs.txt \
    --window_size 1000 \
    --model_type longformer \
    --learning_rate 1e-4 \
    --pad_to_max_length True \
    --save_strategy steps \
    --save_steps 5000 \
    --max_steps 100000 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --seed 42 \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 12 \
    --warmup_steps 10000 \
    --logging_steps 5000 \
    --save_total_limit 20 \
    --output_dir results_nc \
    --tokenizer_name ../data/tokenizer_bpe_9_10_v7/ \
    --config_overrides vocab_size=9,max_position_embeddings=1030,attention_window=32,sep_token_id=1,pad_token_id=3,eos_token_id=1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --adam_epsilon 1e-6 \

#    --tokenizer_name ./tokenizer_unigram_251_v2/ \
#    --max_seq_length 280
#    --per_device_train_batch_size 500 \
#    --per_device_eval_batch_size 500 \
#    --gradient_accumulation_steps 2 \


#    --tokenizer_name ./tokenizer_unigram_4091_v3/ \
#    --config_overrides vocab_size=4096 \
#    --max_seq_length 188 \
#    --per_device_train_batch_size 100 \
#    --per_device_eval_batch_size 100 \
#    --gradient_accumulation_steps 10 \

#    --tokenizer_name ./tokenizer_unigram_251_v2/ \
#    --config_overrides vocab_size=256 \
#    --max_seq_length 280 \
#    --per_device_train_batch_size 64 \
#    --per_device_eval_batch_size 64 \
#    --gradient_accumulation_steps 16 \

#    --resume_from_checkpoint ./results/checkpoint-20000 \
#    --ignore_data_skip

#    --model_type big_bird \
# --config_overrides vocab_size=9,num_random_blocks=1,block_size=16,attention_type=original_full \
