from datasets import load_dataset
import numpy as np
import os
import sys
import tempfile
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments

import plantbert.mlm  # to register auto models
from plantbert.mlm.data_collator_mask_span import DataCollatorForLanguageModelingSpan


N_CPU_WORKERS = 8


# python -i eval_perplexity.py ../../data/mlm/windows/five_prime_UTR.test/512/128/seqs.txt results_512_convnet_ftAth_alone/checkpoint-1000000 
data_path = sys.argv[1]
print("data_path: ", data_path)
model_path = sys.argv[2]
print("model_path: ", model_path)
output_path = f"results/perplexity/{data_path.replace('/', '_')}/{model_path.replace('/', '_')}/"
print("output_path: ", output_path)


dataset = load_dataset("parquet", data_files={"test": data_path})["test"]
#dataset = dataset.select(np.arange(100))
print(dataset)
text_column_name = "seq"

tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize_function(examples):
    res = tokenizer(
        examples[text_column_name],
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    #res["special_tokens_mask"] = np.array(res["special_tokens_mask"]).astype(bool)
    #res["special_tokens_mask"] = res["special_tokens_mask"] | np.char.islower(np.vstack([np.array(list(seq)) for seq in examples[text_column_name]]))
    #print(res["input_ids"][0])
    #print(res["special_tokens_mask"][0])
    #print(res["special_tokens_mask"])
    #print(res["special_tokens_mask"].sum())
    #print(type(res["special_tokens_mask"]))
    #raise Exception("debug")
    return res


dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=N_CPU_WORKERS,
    remove_columns=[text_column_name],
    desc="Running tokenizer on dataset.",
).shuffle(seed=42)
print(dataset)

#dataset = dataset.filter(
#    lambda example: not np.array(example["special_tokens_mask"]).all(),
#    num_proc=N_CPU_WORKERS,
#)
#print(dataset)
#raise Exception("debug")

#dataset = dataset.select([0, 10, 100])
#print(dataset)
#print(dataset["special_tokens_mask"])
#raise Exception("debug")

data_collator = DataCollatorForLanguageModelingSpan(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)

model = AutoModelForMaskedLM.from_pretrained(model_path)

training_args = TrainingArguments(
    output_dir=tempfile.TemporaryDirectory().name,
    per_device_eval_batch_size=256,
    dataloader_num_workers=N_CPU_WORKERS,
    prediction_loss_only=True,
    report_to="none",
    seed=42,
    remove_unused_columns=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset,
    data_collator=data_collator,
)

metrics = trainer.evaluate()

loss = np.array([metrics["eval_loss"]])
perplexity = np.exp(loss)
print("perplexity: ", perplexity)

if not os.path.exists(output_path):
    os.makedirs(output_path)
np.savetxt(os.path.join(output_path, "perplexity.txt"), perplexity)