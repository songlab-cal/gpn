# GPN application to *Arabidopsis thaliana*

## Model usage
Loading the model:
```python
import gpn.model
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-brassicales")
```

## Additional resources
* [GPN logo track at UCSC Genome Browser](https://genome.ucsc.edu/s/gbenegas/gpn-arabidopsis)
* [Training dataset on Hugging Face Hub](https://huggingface.co/datasets/songlab/genomes-brassicales-balanced-v1)
* [Intermediate files necessary for running notebooks, such as embeddings and variant scores](https://huggingface.co/datasets/gonzalobenegas/processed-data-arabidopsis/tree/main)

## Reproducing the analysis
General dependencies:
```bash
conda env create -f envs/general.yaml
conda activate gpn-arabidopsis
```
Analysis code is contained in both `Snakefile` and `*.ipynb`.
