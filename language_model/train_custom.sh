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
    --eval_steps 1000 \
    --seed 42 \
    --fp16 \
    --dataloader_num_workers 8 \
    --warmup_ratio 0.01 \
    --save_total_limit 10 \
    --output_dir results \
    --config_overrides vocab_size=4096 \
    --max_seq_length 188 \
    --per_device_train_batch_size 100 \
    --gradient_accumulation_steps 10 \
    --tokenizer_name ./tokenizer_unigram_4091_v3/ \
#
#    --output_dir results \
#    --config_overrides vocab_size=1024,hidden_size=512,num_hidden_layers=8,num_attention_heads=8,intermediate_size=2048 \
#    --per_device_train_batch_size 120 \
#    --per_device_eval_batch_size 120 \
#    --gradient_accumulation_steps 8 \
#    --tokenizer_name ./tokenizer_unigram_1019_v2/ \
#    --max_seq_length 220 \
#
#
#
#    --resume_from_checkpoint ./results/checkpoint-7848
