import pandas as pd
import sys
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
dataset = pd.read_parquet(f"../datasets/{sys.argv[2]}.parquet")
dataset["tokenized_length"] = [len(s) for s in tokenizer(dataset.seq.values.tolist())["input_ids"]]
print(dataset.tokenized_length.describe())
