# GPN-Star

GPN-Star is a phylogeny-aware genomic language model trained on whole-genome
alignments across multiple evolutionary timescales. The maintained package
supports inference and training on prepared alignments and datasets.

## Quick start

```python
from gpn import register_auto_classes
from transformers import AutoModelForMaskedLM

register_auto_classes("star")
model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-star-hg38-v100-200m")
```

The {doc}`GPN-Star tutorial <../_notebooks/gpn_star_demo>` uses a tiny alignment
fixture. The {doc}`precomputed-score tutorial <../_notebooks/gpn_star_precomputed_scores>`
annotates variants without loading a model or alignment. File-backed inference
uses `gpn star {vep,logits,embedding}`, and the canonical prepared-data training
recipe lives in
[`recipes/gpn_star_training`](https://github.com/songlab-cal/gpn/tree/main/recipes/gpn_star_training).

## Published assets

The [GPN-Star collection](https://huggingface.co/collections/songlab/gpn-star-68c0c055acc2ee51d5c4f129)
accompanies the [bioRxiv paper](https://doi.org/10.1101/2025.09.21.677619).

### Alignments and scores

- [`songlab/gpn-star-scores`](https://huggingface.co/datasets/songlab/gpn-star-scores): genome-wide scores and UCSC Genome Browser tracks.
- [`songlab/multiz100way-pigz`](https://huggingface.co/datasets/songlab/multiz100way-pigz): 100-species vertebrate alignment.
- [`songlab/hg38_cactus447way`](https://huggingface.co/datasets/songlab/hg38_cactus447way): 447-species mammalian alignment.
- [`songlab/mm39_multiz35way`](https://huggingface.co/datasets/songlab/mm39_multiz35way): 35-species mouse alignment.
- [`songlab/galGal6_multiz77way`](https://huggingface.co/datasets/songlab/galGal6_multiz77way): 77-species chicken alignment.
- [`songlab/dm6_multiz124way`](https://huggingface.co/datasets/songlab/dm6_multiz124way): 124-species fly alignment.
- [`songlab/ce11_multiz135way`](https://huggingface.co/datasets/songlab/ce11_multiz135way): 135-species worm alignment.
- [`songlab/tair10_multiz18way`](https://huggingface.co/datasets/songlab/tair10_multiz18way): 18-species arabidopsis alignment.

### Models

Human hg38 checkpoints:

- [`gpn-star-hg38-v100-200m`](https://huggingface.co/songlab/gpn-star-hg38-v100-200m): 100-way vertebrate, 200M parameters.
- [`gpn-star-hg38-m447-200m`](https://huggingface.co/songlab/gpn-star-hg38-m447-200m): 447-way mammalian, 200M parameters.
- [`gpn-star-hg38-p243-200m`](https://huggingface.co/songlab/gpn-star-hg38-p243-200m): 243-way primate, 200M parameters.

Model-organism checkpoints:

- [`gpn-star-mm39-v35-85m`](https://huggingface.co/songlab/gpn-star-mm39-v35-85m): mouse, 85M parameters.
- [`gpn-star-galGal6-v77-85m`](https://huggingface.co/songlab/gpn-star-galGal6-v77-85m): chicken, 85M parameters.
- [`gpn-star-dm6-i124-85m`](https://huggingface.co/songlab/gpn-star-dm6-i124-85m): fly, 85M parameters.
- [`gpn-star-ce11-n135-25m`](https://huggingface.co/songlab/gpn-star-ce11-n135-25m): worm, 25M parameters.
- [`gpn-star-tair10-b18-25m`](https://huggingface.co/songlab/gpn-star-tair10-b18-25m): arabidopsis, 25M parameters.

### Benchmark datasets

Human:

- [`clinvar_vs_benign`](https://huggingface.co/datasets/songlab/clinvar_vs_benign): pathogenic versus benign missense variants.
- [`cosmic`](https://huggingface.co/datasets/songlab/cosmic): frequent COSMIC versus common gnomAD missense variants.
- [`omim_traitgym`](https://huggingface.co/datasets/songlab/omim_traitgym): pathogenic Mendelian regulatory variants versus common variants.
- [`ukb_finemapped_coding`](https://huggingface.co/datasets/songlab/ukb_finemapped_coding): UK Biobank fine-mapped coding variants.
- [`ukb_finemapped_nc_traitgym`](https://huggingface.co/datasets/songlab/ukb_finemapped_nc_traitgym): UK Biobank fine-mapped noncoding variants.
- [`gnomad_balanced`](https://huggingface.co/datasets/songlab/gnomad_balanced): balanced rare-versus-common allele-frequency benchmark.
- [`ldsc`](https://huggingface.co/datasets/songlab/ldsc): S-LDSC variants and predictions for heritability analysis.

Mouse:

- [`wmgp_balanced`](https://huggingface.co/datasets/songlab/wmgp_balanced): Wild Mouse Genome Project allele frequencies.
- [`mmrdb`](https://huggingface.co/datasets/songlab/mmrdb): Mouse Mutant Resource Database pathogenic variants.

Fly:

- [`dest`](https://huggingface.co/datasets/songlab/dest): Drosophila Evolution in Space and Time allele frequencies.
- [`flybase_lethal`](https://huggingface.co/datasets/songlab/flybase_lethal): experimentally validated lethal mutations.

Worm:

- [`caendr`](https://huggingface.co/datasets/songlab/caendr): *C. elegans* Natural Diversity Resource allele frequencies.
- [`celegans_lethal`](https://huggingface.co/datasets/songlab/celegans_lethal): 72 experimentally validated lethal SNVs.

Chicken and arabidopsis:

- [`galbase`](https://huggingface.co/datasets/songlab/galbase): chicken population allele frequencies.
- [`1001gp`](https://huggingface.co/datasets/songlab/1001gp): population allele frequencies from the 1001 Genomes Project.

### Interpretation data

- [`songlab/gpn-star-umap-regions`](https://huggingface.co/datasets/songlab/gpn-star-umap-regions): the labeled hg38 windows used for the published embedding UMAP.

(gpn-star-alignment-data)=
## Alignment data

GPN-Star inference requires a prepared whole-genome alignment that matches the
target assembly, species set, species order, and evolutionary scale used by the
checkpoint. The CLI deliberately does not download these large stores.

### Public human V100 alignment

The public V100 archive is
[`songlab/multiz100way-pigz`](https://huggingface.co/datasets/songlab/multiz100way-pigz).
Its compressed `99.zarr.tar.gz` file is about 42 GB, and the extracted Zarr store
requires additional space. Check both download and extraction capacity before
starting. The demos and test suite use a 3.5 KiB interval fixture and do not need
this archive.

Download the immutable archive and verify it before extraction:

```bash
hf download songlab/multiz100way-pigz 99.zarr.tar.gz \
  --repo-type dataset \
  --revision 6a9d42a35e7debbba845979dea6064f14d5cb3f9 \
  --local-dir .

echo '4dad7da04db9c804032c0c4c7bbefea58f694fc911e962d28c8df87f356ce4ad  99.zarr.tar.gz' \
  | sha256sum --check
unpigz --stdout 99.zarr.tar.gz | tar -xf -
```

The archive extracts as `99.zarr`. Arrange it under the GPN-Star input contract,
where the directory name records the total count including the target species:

```text
/path/to/multiz100way/
└── 100/
    └── all.zarr/  # the extracted 99.zarr store
```

Move or symlink the extracted store to
`/path/to/multiz100way/100/all.zarr`, then pass
`--msa-path /path/to/multiz100way`. A direct
`--msa-path /path/to/multiz100way/100` is also accepted. The logical `100`
directory name is significant even when `all.zarr` is a symlink.

### Inspect a local interval

Querying a short interval is a useful layout check before launching inference:

```python
from gpn.star.data import GenomeMSA

alignment = GenomeMSA(
    "/path/to/multiz100way/100/all.zarr",
    n_species=100,
    in_memory=False,
)
interval = alignment.get_msa(
    "6",
    31_575_665,
    31_575_793,
    strand="+",
    tokenize=False,
)
print(interval.shape)
print(interval[:, 0])
```

Chromosome names must match the store. This example uses the spelling in the
published archive; do not silently add or strip a `chr` prefix. Other GPN-Star
checkpoints can require different public or local alignment stores: check the
model card before substituting one alignment for another.

## Historical analysis

The [GPN-Star paper analysis](https://github.com/songlab-cal/gpn/tree/analysis-archive-2026-08-18/analysis/gpn-star)
is preserved in the historical archive.
