python run_mlm.py \
    --do_train \
    --do_eval \
    --model_type bert \
    --train_file ./lm.train.seqs.txt \
    --preprocessing_num_workers 12 \
    --line_by_line True \
    --pad_to_max_length True \
    --evaluation_strategy epoch \
    --num_train_epochs 100.0 \
    --save_strategy epoch \
    --seed 42 \
    --fp16 \
    --dataloader_num_workers 12 \
    --warmup_ratio 0.01 \
    --save_total_limit 10 \
    --output_dir results \
    --config_overrides vocab_size=256,hidden_size=256,num_hidden_layers=4,num_attention_heads=4,intermediate_size=1024 \
    --per_device_train_batch_size 350 \
    --per_device_eval_batch_size 350 \
    --gradient_accumulation_steps 3 \
    --tokenizer_name ./tokenizer_unigram_251_v2/ \
    --max_seq_length 280 \
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
