# Installation

## Model APIs

Install the package from PyPI:

```bash
pip install gpn
```

This base installation contains the supported model architectures, explicit
AutoClass registration, NumPy, PyTorch, Transformers, and jaxtyping.

## Inference and training

Install file-backed inference commands and their data dependencies with:

```bash
pip install "gpn[inference]"
```

For the maintained GPN and GPN-Star training paths:

```bash
pip install "gpn[train]"
```

PyTorch accelerator selection depends on your platform. For CUDA installations,
follow the [official PyTorch installer](https://pytorch.org/get-started/locally/)
before or alongside GPN.

## Development

GPN supports Python 3.13. Its Transformers version is pinned exactly in the
package metadata, and the complete contributor environment is captured by the
committed uv lockfile:

```bash
git clone https://github.com/songlab-cal/gpn.git
cd gpn
uv sync --extra train --group dev --group docs
uv run pytest
```

Normal tests are offline. Published-checkpoint validation is an explicit,
networked audit and is never scheduled weekly.
