python run_mlm_custom.py \
    --do_train \
    --train_fasta_path ./tair10.train.contigs.fa \
    --do_eval \
    --validation_file ./val_seqs.txt \
    --window_size 1000 \
    --model_type bert \
    --pad_to_max_length True \
    --save_strategy steps \
    --save_steps 2000 \
    --max_steps 20000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --seed 42 \
    --fp16 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 12 \
    --warmup_steps 1000 \
    --logging_steps 1000 \
    --save_total_limit 10 \
    --output_dir results \
    --tokenizer_name ./tokenizer_unigram_251_v2/ \
    --config_overrides vocab_size=256 \
    --max_seq_length 280 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 16 \


#    --tokenizer_name ./tokenizer_unigram_4091_v3/ \
#    --config_overrides vocab_size=4096 \
#    --max_seq_length 188 \
#    --per_device_train_batch_size 100 \
#    --per_device_eval_batch_size 100 \
#    --gradient_accumulation_steps 10 \
