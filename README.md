# GPN (Genomic Pre-trained Network)
[![hgt_genome_392c4_a47ce0](https://github.com/user-attachments/assets/282b6204-156b-4b6d-83ff-2f4a53a9bb2e)](https://genome.ucsc.edu/s/gbenegas/gpn-arabidopsis)
 
Code and resources from [GPN](https://doi.org/10.1073/pnas.2311219120) and related genomic language models.

## Table of contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Modeling frameworks](#modeling-frameworks)
- [Applications of the models](#applications-of-the-models)
- [GPN](#gpn)
- [GPN-MSA](#gpn-msa)
- [Citation](#citation)

## Installation
```bash
pip install git+https://github.com/songlab-cal/gpn.git
```

## Quick start
```python
import gpn.model
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-brassicales")
# or
model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-msa-sapiens")
```

## Modeling frameworks
| Model | Paper | Notes |
| --------- | --- | ----------- |
| GPN | [Benegas et al. 2023](https://doi.org/10.1073/pnas.2311219120) | Requires unaligned genomes | 
| GPN-MSA | [Benegas et al. 2025](https://www.nature.com/articles/s41587-024-02511-w) | Requires aligned genomes for both training and inference |

## Applications of the models
| Paper |  Model | Dataset | Code | Resources on HuggingFace 🤗 |
|  -- | --- | ------- | ---- | -------------- |
| [Benegas et al. 2023](https://doi.org/10.1073/pnas.2311219120) | GPN | Arabidopsis and other Brassicale plants | [analysis/gpn_arabidopsis](analysis/gpn_arabidopsis) |  [Model, dataset, intermediate results](https://huggingface.co/collections/songlab/gpn-653191edcb0270ed05ad2c3e) |
| [Benegas et al. 2025](https://www.nature.com/articles/s41587-024-02511-w) | GPN-MSA | Human and other vertebrates | [analysis/gpn-msa_human](analysis/gpn-msa_human) | [Model, dataset, benchmarks, predictions](https://huggingface.co/collections/songlab/gpn-msa-65319280c93c85e11c803887) |
| [Benegas et al. 2025b](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1) | GPN | Animal promoters | [analysis/gpn_animal_promoter](analysis/gpn_animal_promoter) | [Model, dataset, benchmarks](https://huggingface.co/collections/songlab/traitgym-6796d4fbb825d5b94e65d30f) |
 
## GPN
Can also be called GPN-SS (single sequence).

### Examples
* Play with the model: `examples/ss/basic_example.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songlab-cal/gpn/blob/main/examples/ss/basic_example.ipynb)

### Training on your own data
1. [Snakemake workflow to create a dataset](workflow/make_dataset)
    - Can automatically download data from NCBI given a list of accessions, or use your own fasta files.
2. Training
    - Will automatically detect all available GPUs.
    - Track metrics on [Weights & Biases](https://wandb.ai/)
    - Implemented encoders: `convnet` (default), `roformer` (Transformer), `bytenet`
    - Specify config overrides: e.g. `--config_overrides encoder=bytenet,num_hidden_layers=30`
    - The number of steps that you can train without overfitting will be a function of the size and diversity of your dataset
    - Example:
```bash
WANDB_PROJECT=your_project torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.run_mlm --do_train --do_eval \
    --report_to wandb --prediction_loss_only True --remove_unused_columns False \
    --dataset_name results/dataset --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.1 --soft_masked_loss_weight_evaluation 0.0 \
    --weight_decay 0.01 --optim adamw_torch \
    --dataloader_num_workers 16 --seed 42 \
    --save_strategy steps --save_steps 10000 --evaluation_strategy steps \
    --eval_steps 10000 --logging_steps 10000 --max_steps 120000 --warmup_steps 1000 \
    --learning_rate 1e-3 --lr_scheduler_type constant_with_warmup \
    --run_name your_run --output_dir your_output_dir --model_type GPN \
    --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --gradient_accumulation_steps 1 --total_batch_size 2048 \ 
    --torch_compile \
    --ddp_find_unused_parameters False \
    --bf16 --bf16_full_eval \
```
3. Extract embeddings
    - Input file requires `chrom`, `start`, `end`
    - Example:
```bash
torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.get_embeddings windows.parquet genome.fa.gz 100 your_output_dir \
    results.parquet --per_device_batch_size 4000 --is_file --dataloader_num_workers 16
```
4. Variant effect prediction
    - Input file requires `chrom`, `pos`, `ref`, `alt`
    - Example:
```bash
torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.run_vep variants.parquet genome.fa.gz 512 your_output_dir results.parquet \
    --per_device_batch-size 4000 --is_file --dataloader_num_workers 16
```

## GPN-MSA

### Examples
* Play with the model: `examples/msa/basic_example.ipynb`
* Variant effect prediction: `examples/msa/vep.ipynb`
* Training (human): `examples/msa/training.ipynb`

### Training on other species (e.g. other vertebrates, plants)
* See https://github.com/songlab-cal/gpn/issues/28, https://github.com/songlab-cal/gpn/discussions/40
* Another source for plant alignments: https://plantregmap.gao-lab.org/download.php#alignment-conservation

## Citation
[GPN](https://doi.org/10.1073/pnas.2311219120):
```bibtex
@article{benegas2023dna,
  title={DNA language models are powerful predictors of genome-wide variant effects},
  author={Benegas, Gonzalo and Batra, Sanjit Singh and Song, Yun S},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={44},
  pages={e2311219120},
  year={2023},
  publisher={National Acad Sciences}
}
```

[GPN-MSA](https://www.nature.com/articles/s41587-024-02511-w):
```bibtex
@article{benegas2025dna,
  title={A DNA language model based on multispecies alignment predicts the effects of genome-wide variants},
  author={Benegas, Gonzalo and Albors, Carlos and Aw, Alan J and Ye, Chengzhong and Song, Yun S},
  journal={Nature Biotechnology},
  pages={1--6},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
