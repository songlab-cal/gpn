import os
import numpy as np


data_paths = [
#    "../../data/mlm/windows/five_prime_UTR.test/512/128/seqs.txt",
#    "../../data/mlm/windows/three_prime_UTR.test/512/128/seqs.txt",
#    "../../data/mlm/windows/CDS.test/512/128/seqs.txt",
#    "../../data/mlm/windows/val/512/256/seqs.txt",
    "../../data/mlm/dataset/test/Arabidopsis_thaliana.test.512.256.parquet",
]
#model_paths = [
    #"results_512_convnet/checkpoint-1000000",
    #"results_512_convnet_ftAth_alone/checkpoint-100000",
    #"results_512_convnet_ftAth_alone/checkpoint-500000",
    #"results_512_convnet_ftAth_alone/checkpoint-1000000",   
#]

model_paths = (
    [f"results_512_convnet_only_athaliana/checkpoint-{ckpt}" for ckpt in np.arange(100000, 1100000, 100000)] +
    [f"results_512_convnet_only_athaliana_lower_lr_v2/checkpoint-{ckpt}"  for ckpt in np.arange(20000, 100000, 20000)]
)
print(model_paths)

for data_path in data_paths:
    for model_path in model_paths:
        cmd = f"python eval_perplexity.py {data_path} {model_path}"
        print(cmd)
        os.system(cmd)