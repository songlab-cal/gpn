# GPN (Genomic Pre-trained Network)
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

## Citation
Gonzalo Benegas, Sanjit Singh Batra and Yun S. Song "DNA language models are powerful zero-shot predictors of non-coding variant effects" bioRxiv (2022)  
DOI: [10.1101/2022.08.22.504706](https://doi.org/10.1101/2022.08.22.504706)
