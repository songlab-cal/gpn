python run_mlm_custom.py \
    --do_train \
    --model_type bert \
    --pad_to_max_length True \
    --save_strategy steps \
    --save_steps 2000 \
    --seed 42 \
    --fp16 \
    --dataloader_num_workers 4 \
    --warmup_ratio 0.01 \
    --save_total_limit 10 \
    --output_dir results \
    --config_overrides vocab_size=256,hidden_size=256,num_hidden_layers=4,num_attention_heads=4,intermediate_size=1024 \
    --per_device_train_batch_size 100 \
    --gradient_accumulation_steps 1 \
    --tokenizer_name ./tokenizer_unigram_251_v2/ \
    --max_seq_length 280 \
    --max_steps 10000
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
