# GPN application to *Arabidopsis thaliana*

## Model usage
Loading the model (more details in `basic_example.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songlab-cal/gpn/blob/main/basic_example.ipynb)):
```python
import gpn.model
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("songlab/gpn-brassicales")
```

## Additional resources
- ucsc genome browser
- list of motifs in pdf
- The datasets on huggingface (should probably include the variants)

## Reproducing the analysis
General dependencies:
```bash
conda env create -f envs/general.yaml
conda activate gpn-arabidopsis
```
Analysis code is contained in both `Snakefile` and `*.ipynb`.
