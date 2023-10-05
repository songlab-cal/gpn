# Workflow to create a training dataset
[Example dataset](https://huggingface.co/datasets/gonzalobenegas/example_dataset) (with default config, should take 5 minutes)
1. Download data from NCBI given a list of accessions, or alternatively, use your own fasta files
2. Define a set of training intervals, e.g. full chromosomes, only exons (requires annotation), etc
3. Shard the dataset for efficient loading with Hugging Face libraries
4. Optional: upload to Hugging Face Hub

## Requirements:
- [GPN](https://github.com/songlab-cal/gpn)
- [Snakemake](https://snakemake.github.io/)
- If you want to automatically download data from NCBI, install [NCBI Datasets](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/) (e.g. `conda install -c conda-forge ncbi-datasets-cli`)

## Choosing species/assemblies (ignore if using your own set of fasta files):
- Manually download assembly metadata from [NCBI Genome](https://www.ncbi.nlm.nih.gov/data-hub/genome)
- You can choose a set of taxa (e.g. mammals, plants) and apply filters such as annotation level, assembly level.
- Checkout the script `gpn/ss/filter_assemblies.py` for more details, such as how to
subsample, or how to keep only one assembly per genus.

## Configuration:
- See `config/config.yaml` and `config/assemblies.tsv`
- Check notes in `workflow/Snakefile` for running with your own set of fasta files

## Running:
- `snakemake --cores all`
- The dataset will be created at `results/dataset`

## Uploading to Hugging Face Hub:
For easy distribution and deployment, the dataset can be uploaded to HF Hub (optionally, as a private dataset).
It can be automatically streamed during training (no need to fully download the data locally).
Make sure to first install [HF Hub client library](https://huggingface.co/docs/huggingface_hub/index).
```python
from huggingface_hub import HfApi
api = HfApi()

private = False
repo_id = "gonzalobenegas/example_dataset"  # replace with your username, dataset name
folder_path = "results/dataset"
api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="dataset")
```
