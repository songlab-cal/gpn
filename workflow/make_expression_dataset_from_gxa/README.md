# Workflow to create a gene expression dataset from [Expression Atlas](https://www.ebi.ac.uk/gxa/home)
[Example dataset](https://huggingface.co/datasets/gonzalobenegas/gxa-sorghum-v1) (should take 1 minute to re-create)

1. Download expression count matrices from Expression Atlas given a list of accessions
2. Download genome and annotation from Ensembl
3. Create a dataset of (sequence, expression) pairs
4. Optional: upload to Hugging Face Hub


Main ideas:
- Uses multiple tracks from multiple studies as prediction labels, e.g. different tissues from the same or different studies
- Uses transcript-level TPM, aggregates into TSS bins, and log1p-transforms.
- Groups replicates into "assay groups", filters out groups where replicates don't agree, and averages
- Goal is to have a starting point for further experimentation, e.g. concatenating datasets from multiple species

## Requirements:
```
conda env create -f workflow/envs/general.yaml
conda activate gxa
```

## Choosing studies:
- Go to [Expression Atlas](https://www.ebi.ac.uk/gxa/home), find your species, and find baseline experiments (not tested for differential).
e.g. [list of Sorghum studies](https://www.ebi.ac.uk/gxa/experiments?species=sorghum%20bicolor&experimentType=baseline)
- Pick studies of interest. Check the versions used in the "Supplementary Information" tab.
- Be aware of studies where samples are different genetically from the reference genome - might not be meaningful to train on the reference genome in those cases

## Configuration:
- See `config/config.yaml`
- Most important would be to add the Ensembl url for genome, annotation (try to match expression atlas versions) and the list of studies

## Running:
- `snakemake --cores all`
- The dataset will be created at `results/dataset`

## Uploading to Hugging Face Hub:
For easy distribution and deployment, the dataset can be uploaded to HF Hub (optionally, as a private dataset).
```python
from huggingface_hub import HfApi
api = HfApi()

private = False
repo_id = "gonzalobenegas/gxa-sorghum-v1"  # replace with your username, dataset name
folder_path = "results/dataset"
api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="dataset")
```
