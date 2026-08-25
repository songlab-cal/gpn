# GPN (Genomic Pre-trained Network)
[![hgt_genome_392c4_a47ce0](https://github.com/user-attachments/assets/282b6204-156b-4b6d-83ff-2f4a53a9bb2e)](https://genome.ucsc.edu/s/gbenegas/gpn-arabidopsis)

Code and resources for genomic language models [GPN](https://doi.org/10.1073/pnas.2311219120), [GPN-MSA](https://www.nature.com/articles/s41587-024-02511-w), [PhyloGPN](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7) and [GPN-Star](https://doi.org/10.1101/2025.09.21.677619).

## Table of contents
- [Installation](#installation)
- [Command line](#command-line)
- [Modeling frameworks](#modeling-frameworks)
- [GPN](#gpn)
- [GPN-MSA](#gpn-msa)
- [PhyloGPN](#phylogpn)
- [GPN-Star](#gpn-star)
- [Getting help](#getting-help)
- [Citation](#citation)

## Installation

Install the alpha release from PyPI for the model APIs:

```bash
pip install "gpn==0.9.0a1"
```

Install the dependencies used by the inference CLIs and examples with:

```bash
pip install "gpn[inference]==0.9.0a1"
```

For GPN and GPN-Star training, use the `train` extra:

```bash
pip install "gpn[train]==0.9.0a1"
```

For development, use the locked uv environment:

```bash
git clone https://github.com/songlab-cal/gpn.git
cd gpn
uv sync --all-extras --group dev
```

## Command line

The installed package provides a stable, lazy-loading command:

```text
gpn ss {train,vep,logits,embedding} ...
gpn msa {vep,logits,embedding} ...
gpn star {train,vep,logits,embedding} ...
```

GPN-MSA is inference-only by construction. See the
[CLI guide](docs/cli.md) for inputs, distributed launches, precision controls, and
the maintained command contract.
PhyloGPN and the sorghum gene-expression fine-tune are maintained through explicit
Transformers AutoClass registration and intentionally have no dedicated CLI.

## Modeling frameworks
| Model | Paper | Notes |
| --------- | --- | ----------- |
| GPN | [Benegas et al. 2023](https://doi.org/10.1073/pnas.2311219120) | Requires unaligned genomes |
| GPN-MSA | [Benegas et al. 2025](https://www.nature.com/articles/s41587-024-02511-w) | Deprecated in favor of GPN-Star; inference only |
| PhyloGPN | [Albors et al. 2025](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7) | Inference only; training and fine-tuning are not maintained here |
| GPN-Star | [Ye et al. 2025](https://doi.org/10.1101/2025.09.21.677619) | Requires aligned genomes for both training and inference |

## GPN
A single-sequence genomic language model trained on unaligned genomes. Also known as GPN-SS.

### Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

register_auto_classes("ss")
model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-brassicales")
```

* Play with the model: [examples/ss/basic_example.ipynb](https://github.com/songlab-cal/gpn/blob/main/examples/ss/basic_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songlab-cal/gpn/blob/main/examples/ss/basic_example.ipynb)
* Model implementation: [src/gpn/ss/model.py](https://github.com/songlab-cal/gpn/blob/main/src/gpn/ss/model.py)

### Papers

#### [Benegas, Batra and Song "DNA language models are powerful predictors of genome-wide variant effects" *PNAS* (2023)](https://doi.org/10.1073/pnas.2311219120)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/gpn-653191edcb0270ed05ad2c3e)
- **Pretraining dataset:** Arabidopsis and 7 other Brassicales ([genomes-brassicales-balanced-v1](https://huggingface.co/datasets/songlab/genomes-brassicales-balanced-v1))
- **Models:**
  - [gpn-brassicales](https://huggingface.co/songlab/gpn-brassicales)
- **Historical analysis:**
  - [Brassicales snapshot](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn_arabidopsis)
- **Additional resources:**
  - [processed-data-arabidopsis](https://huggingface.co/datasets/gonzalobenegas/processed-data-arabidopsis)

#### [Benegas, Eraslan and Song "Benchmarking DNA sequence models for causal regulatory variant prediction in human genetics" *bioRxiv* (2025)](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/traitgym-6796d4fbb825d5b94e65d30f)
- **Pretraining dataset:** Animal promoter sequences ([gpn-animal-promoter-dataset](https://huggingface.co/datasets/songlab/gpn-animal-promoter-dataset))
- **Models:**
  - [gpn-animal-promoter](https://huggingface.co/songlab/gpn-animal-promoter)
- **Benchmark datasets:**
  - [TraitGym](https://huggingface.co/datasets/songlab/TraitGym)
- **Historical analysis:**
  - [Animal-promoter snapshot](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn_animal_promoter)
- **Additional resources:**
  - [Checkpoints](https://huggingface.co/datasets/songlab/gpn-animal-promoter-checkpoints)
  - [TraitGym Leaderboard](https://huggingface.co/spaces/songlab/TraitGym-leaderboard)

#### [Groover et al. "Mapping cis-regulatory mutations at scale in sorghum enables modulation of gene expression" *Nature Biotechnology* (2026)](https://www.nature.com/articles/s41587-026-03046-y)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/sorghum-gene-expression-prediction-68963dd31658bfb98c07ae1b)
- **Finetuning dataset:** Sorghum gene expression data from Gene Expression Atlas ([gxa-sorghum-v1](https://huggingface.co/datasets/songlab/gxa-sorghum-v1))
- **Models:**
  - [gpn-brassicales-gxa-sorghum-v1](https://huggingface.co/songlab/gpn-brassicales-gxa-sorghum-v1) (fine-tuned from gpn-brassicales)
- **Historical analysis:**
  - [Sorghum-expression snapshot](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn_sorghum_expression)

### Training on prepared data

The maintained [GPN training recipe](https://github.com/songlab-cal/gpn/tree/main/recipes/gpn_training) starts from
an already prepared sequence dataset and includes tiny CPU and realistic GPU
profiles. Dataset construction is deliberately outside the maintained package.

<details>
<summary><strong>Variant effect prediction</strong></summary>

Input requires `chrom`, one-based `pos`, and distinct uppercase canonical SNV
`ref`/`alt` columns. A mismatch between `ref` and the local genome is an error.

Example command:
```bash
gpn ss vep \
    --input-path variants.parquet \
    --genome-path genome.fa.gz \
    --window-size 512 \
    --model-path songlab/gpn-brassicales \
    --output-path results.parquet \
    --per-device-eval-batch-size 64 \
    --dataloader-num-workers 8 \
    --bf16-full-eval \
    --torch-compile
```

Local table inputs are detected automatically. This quick start uses one GPU;
the [CLI guide](docs/cli.md#devices-and-distributed-execution) also shows
multi-GPU inference with `torchrun`.

</details>

## GPN-MSA
A deprecated genomic language model trained on whole-genome alignments across
multiple species. The published checkpoint remains supported for inference;
training code and workflows are preserved only in the historical archive.

### Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

register_auto_classes("msa")
model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-msa-sapiens")
```

* Model implementation: [src/gpn/msa/model.py](https://github.com/songlab-cal/gpn/blob/main/src/gpn/msa/model.py)
* Scientific baseline: [published-model fixtures](https://github.com/songlab-cal/gpn/tree/main/tests/fixtures)

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
- **Historical analysis and retired notebooks:**
  - [GPN-MSA snapshot](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn-msa_human)
- **Additional resources:**
  - [hg38 genome-wide scores](https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores)
  - [Gene essentiality predictions](https://huggingface.co/datasets/songlab/gpn-msa-hg38-gene-essentiality-scores)

## PhyloGPN
A phylogenetic genomic language model trained on the Zoonomia alignment.
PhyloGPN is a convolutional neural network that outputs rate-matrix parameters
for Felsenstein's F81 substitution model. The published checkpoint is maintained
here for inference, including zero-shot variant-effect prediction for sequences
outside reference genomes.

### Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModel

register_auto_classes("phylo")
model = AutoModel.from_pretrained("songlab/PhyloGPN")
```

PhyloGPN inference is supported through this explicit AutoClass API; it does not
have a dedicated `gpn` CLI command.

* Play with the model: [examples/phylogpn/basic_example.ipynb](https://github.com/songlab-cal/gpn/blob/main/examples/phylogpn/basic_example.ipynb)
* Model implementation: [src/gpn/phylo/model.py](https://github.com/songlab-cal/gpn/blob/main/src/gpn/phylo/model.py)

### Papers

#### [Albors, Li, Benegas, Ye and Song "A Phylogenetic Approach to Genomic Language Modeling" *RECOMB* (2025)](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7)

- **Models:**
  - [PhyloGPN](https://huggingface.co/songlab/PhyloGPN)

## GPN-Star
A phylogeny-aware genomic language model trained on whole-genome alignments across multiple evolutionary timescales.

### Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

register_auto_classes("star")
model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-star-hg38-p243-200m")
```

* Play with the model: [examples/star/demo.ipynb](https://github.com/songlab-cal/gpn/blob/main/examples/star/demo.ipynb)
* Model implementation: [src/gpn/star/model.py](https://github.com/songlab-cal/gpn/blob/main/src/gpn/star/model.py)
* Training on prepared data: [GPN-Star recipe](https://github.com/songlab-cal/gpn/tree/main/recipes/gpn_star_training)

### Papers

#### [Ye, Benegas, Albors, Li, Prillo, Fields, Clarke and Song "Predicting functional constraints across evolutionary timescales with phylogeny-informed genomic language models" *bioRxiv* (2025)](https://doi.org/10.1101/2025.09.21.677619)

- **Collection:** [HuggingFace 🤗](https://huggingface.co/collections/songlab/gpn-star-68c0c055acc2ee51d5c4f129)
- **Genome-wide scores and UCSC Genome Browser tracks:** [gpn-star-scores](https://huggingface.co/datasets/songlab/gpn-star-scores)
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
- **Historical analysis:**
  - [GPN-Star snapshot](https://github.com/songlab-cal/gpn/tree/30dee6cf45849dfdcfc043ca8baf44fd6ba51d74/analysis/gpn-star)

## Getting help

- **Questions?** Open a [Discussion](https://github.com/songlab-cal/gpn/discussions) for usage questions, ideas, or general help
- **Issues?** Report bugs or request features via [Issues](https://github.com/songlab-cal/gpn/issues)
- **Research code?** See the [research branch and archive policy](https://github.com/songlab-cal/gpn/blob/main/docs/research.md)

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

[Sorghum gene expression prediction](https://www.nature.com/articles/s41587-026-03046-y):
```bibtex
@article{groover2026mapping,
  title={Mapping cis-regulatory mutations at scale in sorghum enables modulation of gene expression},
  author={Groover, Evan D and Ding, David and Wang, Flora Z and Benegas, Gonzalo and Rivera, Joseph and Schwartz, Shahar and Chen, Stephen and Moubarak, Michael F and Georgieva, Viktoriya and Lemaux, Peggy G and others},
  journal={Nature Biotechnology},
  pages={1--11},
  year={2026},
  publisher={Nature Publishing Group US New York}
}
```
