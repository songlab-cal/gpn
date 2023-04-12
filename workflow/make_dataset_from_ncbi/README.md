# Workflow to create a training dataset for any set of taxa
For example usage, check out `analysis/arabidopsis/Snakefile` and `analysis/arabidopsis/config.yaml`.

As a quick preview, here's how you could integrate this as a Snakemake sub-workflow:
```python
rule all:
    input:
        expand(f"output/merged_dataset/{config['window_size']}/{config['step_size']}/{config['add_rc']}/balanced/data/{{split}}", split=splits),


module make_dataset_from_ncbi:
    snakefile:
        "https://raw.githubusercontent.com/songlab-cal/gpn/main/workflow/make_dataset_from_ncbi/Snakefile"
    config: config


use rule * from make_dataset_from_ncbi as make_dataset_from_ncbi_*
```
