# GPN (Genomic Pre-trained Network)
[![hgt_genome_392c4_a47ce0](https://github.com/user-attachments/assets/282b6204-156b-4b6d-83ff-2f4a53a9bb2e)](https://genome.ucsc.edu/s/gbenegas/gpn-arabidopsis)
 
Code and resources for genomic language models [GPN](https://doi.org/10.1073/pnas.2311219120), [GPN-MSA](https://www.nature.com/articles/s41587-024-02511-w), [PhyloGPN](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7) and [GPN-Star](https://doi.org/10.1101/2025.09.21.677619).

## Table of contents
- [Installation](#installation)
- [Modeling frameworks](#modeling-frameworks)
- [GPN](#gpn)
- [GPN-MSA](#gpn-msa)
- [PhyloGPN](#phylogpn)
- [GPN-Star](#gpn-star)
- [Getting help](#getting-help)
- [Citation](#citation)

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/songlab-cal/gpn.git
```

For development (editable install):

```bash
git clone https://github.com/songlab-cal/gpn.git
cd gpn
pip install -e .
```

## Modeling frameworks
| Model | Paper | Notes |
| --------- | --- | ----------- |
| GPN | [Benegas et al. 2023](https://doi.org/10.1073/pnas.2311219120) | Requires unaligned genomes | 
| GPN-MSA | [Benegas et al. 2025](https://www.nature.com/articles/s41587-024-02511-w) | Requires aligned genomes for both training and inference [deprecated in favor of GPN-Star] |
| PhyloGPN | [Albors et al. 2025](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7) | Uses an alignment during training, but does not require it for inference or fine-tuning |
| GPN-Star | [Ye et al. 2025](https://doi.org/10.1101/2025.09.21.677619) | Requires aligned genomes for both training and inference |

## GPN
A single-sequence genomic language model trained on unaligned genomes. Also known as GPN-SS.

### Quick start

```python
import gpn.model  # registers architecture for AutoModel
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-brassicales")
```

* Play with the model: [examples/ss/basic_example.ipynb](examples/ss/basic_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songlab-cal/gpn/blob/main/examples/ss/basic_example.ipynb)
* Model implementation: [gpn/model.py](gpn/model.py), [gpn/ss](gpn/ss)

### Papers

#### [Benegas, Batra and Song "DNA language models are powerful predictors of genome-wide variant effects" *PNAS* (2023)](https://doi.org/10.1073/pnas.2311219120)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/gpn-653191edcb0270ed05ad2c3e)
- **Pretraining dataset:** Arabidopsis and 7 other Brassicales ([genomes-brassicales-balanced-v1](https://huggingface.co/datasets/songlab/genomes-brassicales-balanced-v1))
- **Models:**
  - [gpn-brassicales](https://huggingface.co/songlab/gpn-brassicales)
- **Analysis code:**
  - [analysis/gpn_arabidopsis](analysis/gpn_arabidopsis)
- **Additional resources:**
  - [processed-data-arabidopsis](https://huggingface.co/datasets/gonzalobenegas/processed-data-arabidopsis)

#### [Benegas, Eraslan and Song "Benchmarking DNA sequence models for causal regulatory variant prediction in human genetics" *bioRxiv* (2025)](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/traitgym-6796d4fbb825d5b94e65d30f)
- **Pretraining dataset:** Animal promoter sequences ([gpn-animal-promoter-dataset](https://huggingface.co/datasets/songlab/gpn-animal-promoter-dataset))
- **Models:**
  - [gpn-animal-promoter](https://huggingface.co/songlab/gpn-animal-promoter)
- **Benchmark datasets:**
  - [TraitGym](https://huggingface.co/datasets/songlab/TraitGym)
- **Analysis code:**
  - [analysis/gpn_animal_promoter](analysis/gpn_animal_promoter)
- **Additional resources:**
  - [Checkpoints](https://huggingface.co/datasets/songlab/gpn-animal-promoter-checkpoints)
  - [TraitGym Leaderboard](https://huggingface.co/spaces/songlab/TraitGym-leaderboard)

#### Sorghum gene expression prediction (unpublished)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/sorghum-gene-expression-prediction-68963dd31658bfb98c07ae1b)
- **Finetuning dataset:** Sorghum gene expression data from Gene Expression Atlas ([gxa-sorghum-v1](https://huggingface.co/datasets/songlab/gxa-sorghum-v1))
- **Models:**
  - [gpn-brassicales-gxa-sorghum-v1](https://huggingface.co/songlab/gpn-brassicales-gxa-sorghum-v1) (fine-tuned from gpn-brassicales)
- **Analysis code:**
  - [analysis/gpn_sorghum_expression](analysis/gpn_sorghum_expression)

### Training on your own data

<details>
<summary><strong>1. Create a dataset</strong></summary>

Use the [Snakemake workflow](workflow/make_dataset) to create a dataset:
- Can automatically download data from NCBI given a list of accessions, or use your own fasta files
- Navigate to `workflow/make_dataset/`, configure `config/config.yaml` and `config/assemblies.tsv`, then run:
  ```bash
  snakemake --cores all
  ```

</details>

<details>
<summary><strong>2. Train the model</strong></summary>

Training features:
- Automatically detects all available GPUs
- Track metrics on [Weights & Biases](https://wandb.ai/)
- Implemented encoders: `convnet` (default), `roformer` (Transformer), `bytenet`
- Specify config overrides: e.g. `--config_overrides encoder=bytenet,num_hidden_layers=30`
- The number of steps that you can train without overfitting will be a function of the size and diversity of your dataset

Example command:
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
    --bf16 --bf16_full_eval
```

</details>

<details>
<summary><strong>3. Extract embeddings</strong></summary>

Input file requires `chrom`, `start`, `end` columns.

Example command:
```bash
torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.get_embeddings \
    windows.parquet genome.fa.gz 100 your_output_dir results.parquet \
    --per_device_batch_size 4000 --is_file --dataloader_num_workers 16
```

</details>

<details>
<summary><strong>4. Variant effect prediction</strong></summary>

Input file requires `chrom`, `pos`, `ref`, `alt` columns.

Example command:
```bash
torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.run_vep \
    variants.parquet genome.fa.gz 512 your_output_dir results.parquet \
    --per_device_batch_size 4000 --is_file --dataloader_num_workers 16
```

</details>

## GPN-MSA
A genomic language model trained on whole-genome alignments across multiple species.

### Quick start

```python
import gpn.model  # registers architecture for AutoModel
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-msa-sapiens")
```

* Play with the model: [examples/msa/basic_example.ipynb](examples/msa/basic_example.ipynb)
* Variant effect prediction: [examples/msa/vep.ipynb](examples/msa/vep.ipynb)
* Training (human): [examples/msa/training.ipynb](examples/msa/training.ipynb)
* Model implementation: [gpn/model.py](gpn/model.py), [gpn/msa](gpn/msa)

### Papers

#### [Benegas, Albors, Aw, Ye and Song "A DNA language model based on multispecies alignment predicts the effects of genome-wide variants" *Nature Biotechnology* (2025)](https://www.nature.com/articles/s41587-024-02511-w)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/gpn-msa-65319280c93c85e11c803887)
- **Pretraining datasets:**
  - 100-way vertebrate alignment: [multiz100way](https://huggingface.co/datasets/songlab/multiz100way-pigz), [89.zarr](https://huggingface.co/datasets/lpigou/89.zarr)
  - Training regions: [gpn-msa-sapiens-dataset](https://huggingface.co/datasets/songlab/gpn-msa-sapiens-dataset)
- **Models:**
  - [gpn-msa-sapiens](https://huggingface.co/songlab/gpn-msa-sapiens)
- **Benchmark datasets (including predictions from all models):**
  - [ClinVar](https://huggingface.co/datasets/songlab/clinvar) - Missense variants with clinical pathogenic/benign labels
  - [COSMIC](https://huggingface.co/datasets/songlab/cosmic) - Somatic missense mutations in cancer
  - [OMIM](https://huggingface.co/datasets/songlab/omim) - Regulatory variants implicated in Mendelian disorders
  - [gnomAD](https://huggingface.co/datasets/songlab/gnomad) - Genome-wide variants with allele frequency information
- **Analysis code:**
  - [analysis/gpn-msa_human](analysis/gpn-msa_human)
- **Additional resources:**
  - [hg38 genome-wide scores](https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores)
  - [Gene essentiality predictions](https://huggingface.co/datasets/songlab/gpn-msa-hg38-gene-essentiality-scores)

### Training on other species (e.g. other vertebrates, plants)
* See https://github.com/songlab-cal/gpn/issues/28, https://github.com/songlab-cal/gpn/discussions/40, https://github.com/songlab-cal/gpn/issues/44
* Another source for plant alignments: https://plantregmap.gao-lab.org/download.php#alignment-conservation

## PhyloGPN
A phylogenetic genomic language model that uses an alignment during training but does not require it for inference or fine-tuning. PhyloGPN is a convolutional neural network that outputs rate matrix parameters for Felsenstein's F81 substitution model, trained on the Zoonomia alignment. It can be used for transfer learning and zero-shot variant deleteriousness prediction, especially useful for sequences not in reference genomes.

### Quick start

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("songlab/PhyloGPN", trust_remote_code=True)
```

* Play with the model: [examples/phylogpn/basic_example.ipynb](examples/phylogpn/basic_example.ipynb)
* Model implementation: [gpn/phylogpn.py](gpn/phylogpn.py)

### Papers

#### [Albors, Li, Benegas, Ye and Song "A Phylogenetic Approach to Genomic Language Modeling" *RECOMB* (2025)](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7)

- **Models:**
  - [PhyloGPN](https://huggingface.co/songlab/PhyloGPN)

## GPN-Star
A phylogeny-aware genomic language model trained on whole-genome alignments across multiple evolutionary timescales.

### Quick start

```python
import gpn.star.model  # registers architecture for AutoModel
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-star-hg38-p243-200m")
```

* Play with the model: [examples/star/demo.ipynb](examples/star/demo.ipynb)
* Model implementation: [gpn/model.py](gpn/model.py), [gpn/star](gpn/star)

### Papers

#### [Ye, Benegas, Albors, Li, Prillo, Fields, Clarke and Song "Predicting functional constraints across evolutionary timescales with phylogeny-informed genomic language models" *bioRxiv* (2025)](https://doi.org/10.1101/2025.09.21.677619)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/gpn-star-68c0c055acc2ee51d5c4f129)
- **Pretraining datasets:**
  - Vertebrate alignment: [multiz100way](https://huggingface.co/datasets/songlab/multiz100way-pigz) (100 species)
  - Mammalian alignment: [cactus447way](https://huggingface.co/datasets/songlab/hg38_cactus447way) (447 species)
- **Models:**
  - Human (hg38):
    - [gpn-star-hg38-v100-200m](https://huggingface.co/songlab/gpn-star-hg38-v100-200m) (vertebrate, 200M params)
    - [gpn-star-hg38-m447-200m](https://huggingface.co/songlab/gpn-star-hg38-m447-200m) (mammalian, 200M params)
    - [gpn-star-hg38-p243-200m](https://huggingface.co/songlab/gpn-star-hg38-p243-200m) (primate, 200M params)
  - Model organisms:
    - [gpn-star-mm39-v35-85m](https://huggingface.co/songlab/gpn-star-mm39-v35-85m) (mouse, 85M params)
    - [gpn-star-galGal6-v77-85m](https://huggingface.co/songlab/gpn-star-galGal6-v77-85m) (chicken, 85M params)
    - [gpn-star-dm6-i124-85m](https://huggingface.co/songlab/gpn-star-dm6-i124-85m) (fly, 85M params)
    - [gpn-star-ce11-n135-25m](https://huggingface.co/songlab/gpn-star-ce11-n135-25m) (worm, 25M params)
    - [gpn-star-tair10-b18-25m](https://huggingface.co/songlab/gpn-star-tair10-b18-25m) (arabidopsis, 25M params)
- **Benchmark datasets (including predictions from all models):**
  - Included in [collection](https://huggingface.co/collections/songlab/gpn-star-68c0c055acc2ee51d5c4f129)
  - **Homo sapiens:**
    - [clinvar_vs_benign](https://huggingface.co/datasets/songlab/clinvar_vs_benign) - Missense variant pathogenicity classification (Pathogenic vs. Benign)
    - [cosmic](https://huggingface.co/datasets/songlab/cosmic) - Cancer somatic mutations (COSMIC frequent vs. gnomAD common missense)
    - [omim_traitgym](https://huggingface.co/datasets/songlab/omim_traitgym) - Mendelian regulatory variants (pathogenic vs. common)
    - [ukb_finemapped_coding](https://huggingface.co/datasets/songlab/ukb_finemapped_coding) - UK Biobank fine-mapped coding variants
    - [ukb_finemapped_nc_traitgym](https://huggingface.co/datasets/songlab/ukb_finemapped_nc_traitgym) - UK Biobank fine-mapped non-coding variants
    - [gnomad_balanced](https://huggingface.co/datasets/songlab/gnomad_balanced) - Allele frequency enrichment analysis (rare vs. common variants)
    - [ldsc](https://huggingface.co/datasets/songlab/ldsc) - S-LDSC variants and model predictions for heritability analysis
  - **Mus musculus:**
    - [wmgp_balanced](https://huggingface.co/datasets/songlab/wmgp_balanced) - Wild Mouse Genome Project population allele frequencies
    - [mmrdb](https://huggingface.co/datasets/songlab/mmrdb) - Mouse Mutant Resource Database pathogenic variants
  - **Drosophila melanogaster:**
    - [dest](https://huggingface.co/datasets/songlab/dest) - Drosophila Evolution in Space and Time allele frequencies
    - [flybase_lethal](https://huggingface.co/datasets/songlab/flybase_lethal) - Experimentally validated lethal mutations
  - **Caenorhabditis elegans:**
    - [caendr](https://huggingface.co/datasets/songlab/caendr) - C. elegans Natural Diversity Resource allele frequencies
    - [celegans_lethal](https://huggingface.co/datasets/songlab/celegans_lethal) - 72 experimentally validated lethal SNVs
  - **Gallus gallus:**
    - [galbase](https://huggingface.co/datasets/songlab/galbase) - Chicken population allele frequencies
  - **Arabidopsis thaliana:**
    - [1001gp](https://huggingface.co/datasets/songlab/1001gp) - Population allele frequencies from 1001 Genome Project
- **Analysis code:**
  - Model training and main results on variant effect prediction: [analysis/gpn-star/train_and_eval](analysis/gpn-star/train_and_eval)
  - Complex trait heritability analysis (S-LDSC): [analysis/gpn-star/s-ldsc](analysis/gpn-star/s-ldsc)
  - Whole-genome alignment processing: [analysis/gpn-star/wga_processing](analysis/gpn-star/wga_processing)
  - Model interpretation: [analysis/gpn-star/interpretation](analysis/gpn-star/interpretation)

## Getting help

- **Questions?** Open a [Discussion](https://github.com/songlab-cal/gpn/discussions) for usage questions, ideas, or general help
- **Issues?** Report bugs or request features via [Issues](https://github.com/songlab-cal/gpn/issues)

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

[PhyloGPN](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7):
```bibtex
@inproceedings{albors2025phylogenetic,
  title={A Phylogenetic Approach to Genomic Language Modeling},
  author={Albors, Carlos and Li, Jianan Canal and Benegas, Gonzalo and Ye, Chengzhong and Song, Yun S},
  booktitle={International Conference on Research in Computational Molecular Biology},
  pages={99--117},
  year={2025},
  organization={Springer}
}
```

[GPN-Star](https://doi.org/10.1101/2025.09.21.677619):
```bibtex
@article{ye2025predicting,
  title={Predicting functional constraints across evolutionary timescales with phylogeny-informed genomic language models},
  author={Ye, Chengzhong and Benegas, Gonzalo and Albors, Carlos and Li, Jianan Canal and Prillo, Sebastian and Fields, Peter D and Clarke, Brian and Song, Yun S},
  journal={bioRxiv},
  pages={2025--09},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
