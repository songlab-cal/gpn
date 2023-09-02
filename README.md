# GPN (Genomic Pre-trained Network)
[![hgt_genome_392c4_a47ce0](https://user-images.githubusercontent.com/5766420/228109137-85d48559-d1ae-4c9a-94b5-c79fc06ad45d.png)](  https://genome.ucsc.edu/s/gbenegas/gpn-arabidopsis)


## Installation
```bash
pip install git+https://github.com/songlab-cal/gpn.git
```

## Application to *Arabidopsis thaliana*
* Quick example to play with the model: `basic_example.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songlab-cal/gpn/blob/main/basic_example.ipynb)
* [Training, inference and analysis](analysis/arabidopsis)

## Training on your own data
1. [Snakemake workflow to create a dataset](workflow/make_dataset)
    - Can automatically download data from NCBI given a list of accessions, or use your own fasta files.
2. Training
    - Will automatically detect all available GPUs.
    - Track metrics on [Weights & Biases](https://wandb.ai/)
    - Implemented models: `ConvNet`, `GPNRoFormer` (Transformer)
    - Specify config overrides: e.g. `--config_overrides n_layers=30`
    - Example:
```bash
WANDB_PROJECT=your_project python -m gpn.run_mlm --do_train --do_eval \
    --fp16 --report_to wandb --prediction_loss_only True --remove_unused_columns False \
    --dataset_name results/dataset --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 --optim adamw_torch \
    --dataloader_num_workers 16 --seed 42 \
    --save_strategy steps --save_steps 10000 --evaluation_strategy steps \
    --eval_steps 10000 --logging_steps 10000 --max_steps 120000 --warmup_steps 1000 \
    --learning_rate 1e-3 --lr_scheduler_type constant_with_warmup \
    --run_name your_run --output_dir your_output_dir --model_type ConvNet \
    --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --gradient_accumulation_steps 1
```
3. Extract embeddings
    - Input file requires `chrom`, `start`, `end`
    - Example:
```bash
python -m gpn.get_embeddings windows.parquet genome.fa.gz 100 your_output_dir \
    results.parquet --per-device-batch-size 4000 --is-file --dataloader-num-workers 16
```
4. Variant effect prediction
    - Input file requires `chrom`, `pos`, `ref`, `alt`
    - Example:
```bash
python -m gpn.run_vep variants.parquet genome.fa.gz 512 your_output_dir results.parquet \
    --per-device-batch-size 4000 --is-file --dataloader-num-workers 16
```

## Citation
```
@article{benegas2023dna,
	author = {Gonzalo Benegas and Sanjit Singh Batra and Yun S. Song},
	title = {DNA language models are powerful predictors of genome-wide variant effects},
	elocation-id = {2022.08.22.504706},
	year = {2023},
	doi = {10.1101/2022.08.22.504706},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/08/04/2022.08.22.504706},
	eprint = {https://www.biorxiv.org/content/early/2023/08/04/2022.08.22.504706.full.pdf},
	journal = {bioRxiv}
}
```
