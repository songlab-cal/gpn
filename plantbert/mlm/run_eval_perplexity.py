import os

data_paths = [
    "../../data/mlm/windows/five_prime_UTR.test/512/128/seqs.txt",
    "../../data/mlm/windows/three_prime_UTR.test/512/128/seqs.txt",
    "../../data/mlm/windows/CDS.test/512/128/seqs.txt",
    "../../data/mlm/windows/val/512/256/seqs.txt",
]
model_paths = [
    "results_512_convnet/checkpoint-1000000",
    "results_512_convnet_ftAth_alone/checkpoint-100000",
    "results_512_convnet_ftAth_alone/checkpoint-500000",
    "results_512_convnet_ftAth_alone/checkpoint-1000000",
]

for data_path in data_paths:
    for model_path in model_paths:
        cmd = f"python eval_perplexity.py {data_path} {model_path}"
        print(cmd)
        os.system(cmd)