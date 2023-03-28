# GPN application to *Arabidopsis thaliana*

## Model usage
Loading the model:
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
