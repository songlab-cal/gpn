# GPN-Promoter training (second part)

## Setup

```bash
# install software
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

# download dataset
hf download songlab/gpn-animal-promoter-dataset --repo-type dataset --local-dir dataset/
```

## Running

```bash
source .venv/bin/activate
bash train.sh dataset checkpoints 128 2
```
