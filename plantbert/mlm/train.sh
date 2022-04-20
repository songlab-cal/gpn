python ./run_mlm.py \
    --do_train \
    --train_fasta_path ../data/genomes/all.contigs.fa.gz \
    --do_eval \
    --validation_file ../data/windows/val/1000/100/seqs.txt \
    --window_size 1000 \
    --model_type bert \
    --learning_rate 6e-4 \
    --pad_to_max_length True \
    --save_strategy steps \
    --save_steps 10000 \
    --max_steps 200000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --warmup_steps 10000 \
    --logging_steps 10000 \
    --save_total_limit 20 \
    --output_dir results \
    --tokenizer_name ../data/tokenizer_bpe_8192_v5/ \
    --config_overrides vocab_size=8192 \
    --max_seq_length 200 \
    --per_device_train_batch_size 170 \
    --per_device_eval_batch_size 170 \
    --gradient_accumulation_steps 3 \
    --fp16 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --adam_epsilon 1e-4 \
    --seed 44 \
    --resume_from_checkpoint ./results/checkpoint-100000 \
    --ignore_data_skip

#    --tokenizer_name ../data/tokenizer_unigram_8192_50000_v5/ \
#    --config_overrides vocab_size=8192 \
#    --max_seq_length 170 \
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
