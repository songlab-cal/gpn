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
* [Promoter motifs obtained with GPN + TF-MoDISco](promoter_motifs_gpn_modisco.pdf)
* [Training dataset on Hugging Face Hub](https://huggingface.co/datasets/songlab/genomes-brassicales-balanced-v1)

## Reproducing the analysis
General dependencies:
```bash
conda env create -f envs/general.yaml
conda activate gpn-arabidopsis
```
Analysis code is contained in both `Snakefile` and `*.ipynb`.
