# S-LDSC

## Requirements

TODO: provide an environment

Optional (recommended): download the processed data:
- S-LDSC reference files, originally in (https://zenodo.org/records/10515792) but packaged here as well for your convenience.
- Models
- Traits
- Results (models x traits)
```bash
# first install https://huggingface.co/docs/huggingface_hub/en/guides/cli
mkdir -p results
hf download songlab/ldsc --repo-type dataset --local-dir results/
```

## Running


Take a look at `workflow/Snakefile` rule `all` for example targets.
The first one will run S-LDSC on one model on one trait.

```python
# if you use mamba:
snakemake --cores all --use-conda --conda-frontend mamba
# else:
snakemake --cores all --use-conda
```

```bash
# Snakemake sometimes gets confused about which files it needs to rerun and this forces
# not to rerun any existing file
snakemake --cores all --touch
# to output an execution plan
snakemake --cores all --dry-run
```

LDSC jobs (e.g. running model X on trait Y) is by default parallelized as 1 job per core.
This works as long as you have enough memory (e.g. when asking for a complete `savio3_htc` or `savio4_htc` node).
When running on a node with less memory, should reduce the parallelization of the `ld_score` and `run_ldsc_annot` rules in `workflow/rules/ldsc.smk`.


To add a new model `{model}`, place a parquet file with column `score` in either of these locations:
- `results/variant_scores/{model}.parquet`

    Variants should be in the standard S-LDSC order
    (e.g. see the 9,997,231 variants in https://huggingface.co/datasets/songlab/ldsc)
- `results/features/{model}.parquet`

    This corresponds to the variants above with `pos` != -1.
    `pos` == -1 are a tiny fraction where we were not able to liftover from `hg19` to `hg38`.
    This complexity is because S-LDSC uses `hg19` but we do most our work with newer annotations in `hg38`.

To add a new trait `{trait}`, place it under `results/sumstats_107/{trait}.sumstats.gz` (please check format of other traits).