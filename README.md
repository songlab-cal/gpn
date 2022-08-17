# GPN (Genomic Pretrained Network)
## Installation
```bash
pip install git+https://github.com/songlab-cal/gpn.git
```
## Usage
Loading the model (more details in `basic_example.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/songlab-cal/gpn/blob/main/basic_example.ipynb)):
```python
import gpn.mlm
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("gonzalobenegas/gpn-arabidopsis")
```
Example scripts for different tasks:
- Preparing data: `data/mlm/Snakefile`
- Training: `analysis/mlm/train_512_convnet_only_athaliana.sh`
- Variant effect prediction: `analysis/mlm/run_vep.py`
