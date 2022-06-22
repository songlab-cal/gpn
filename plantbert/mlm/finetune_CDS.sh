WANDB_PROJECT=PlantBERT_MLM_CDS python ./run_mlm_custom.py \
    --report_to wandb \
    --run_name ConvNet \
    --do_train \
    --do_eval \
    --train_fasta_path ../../data/mlm/dataset/CDS.parquet \
    --validation_file ../../data/mlm/windows/CDS.test/512/128/seqs.txt \
    --model_name_or_path ./results_512_convnet_finetuning_v2/checkpoint-1000000/ \
    --model_type ConvNet \
    --line_by_line True \
    --window_size 512 \
    --learning_rate 1e-3 \
    --save_strategy steps \
    --save_steps 10000 \
    --max_steps 20000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --warmup_steps 1000 \
    --logging_steps 1000 \
    --output_dir results_CDS_ConvNet \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --adam_epsilon 1e-4 \
    --seed 42 \
    --prediction_loss_only True \
    --lr_scheduler_type constant_with_warmup \
#    --resume_from_checkpoint ./results_512_convnet/checkpoint-600000 \
#    --ignore_data_skip \

#    --overwrite_cache True \
#    --save_total_limit 10 \



#    --eval_accumulation_steps 50 \

#    --train_fasta_path ../../data/mlm/genomes/Arabidopsis_thaliana_train.contigs.fa.gz \
#--train_fasta_path ../../data/mlm/genomes/all.contigs.fa.gz \
#    --overwrite_cache True \

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
