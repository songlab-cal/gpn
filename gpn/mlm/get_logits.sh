#python get_logits.py ./results_512_convnet_finetuning_v2/checkpoint-1000000/ ./examples_Chr5:3687412-3688325.parquet logits_Chr5:3687412-3688325.parquet
#python get_logits.py ./results_512_convnet_finetuning_v2/checkpoint-1000000/ ./examples_Chr5:3564493-3565087.parquet logits_Chr5:3564493-3565087.parquet
#python get_logits.py ./results_512_convnet_finetuning_v2/checkpoint-1000000/ ./examples_Chr5:552887-553012.parquet logits_Chr5:552887-553012.parquet
#python get_logits.py ./results_512_convnet_finetuning_v2/checkpoint-1000000/ ./examples_Chr5:3500000-3600000.parquet logits_Chr5:3500000-3600000.parquet
python get_logits.py results_512_convnet_only_athaliana_lower_lr_v2/checkpoint-80000 ./examples_Chr5:3500000-3600000.parquet results/logits/logits_Chr5:3500000-3600000.parquet
