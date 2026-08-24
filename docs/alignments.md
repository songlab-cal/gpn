# GPN-Star alignment data

GPN-Star inference requires a prepared whole-genome alignment that matches the
target assembly, species set, species order, and evolutionary scale used by the
checkpoint. The CLI deliberately does not download these large stores.

## Public human V100 alignment

The public V100 archive is
[`songlab/multiz100way-pigz`](https://huggingface.co/datasets/songlab/multiz100way-pigz).
Its compressed `99.zarr.tar.gz` file is about 42 GB, and the extracted Zarr store
requires additional space. Check both the download and extraction capacity before
starting. The demos and test suite use a 3.5 KiB interval fixture and do not need
this archive.

Download the immutable archive and verify it before extraction:

```bash
hf download songlab/multiz100way-pigz 99.zarr.tar.gz \
  --repo-type dataset \
  --revision 6a9d42a35e7debbba845979dea6064f14d5cb3f9 \
  --local-dir .

echo '4dad7da04db9c804032c0c4c7bbefea58f694fc911e962d28c8df87f356ce4ad  99.zarr.tar.gz' \
  | sha256sum --check
unpigz --stdout 99.zarr.tar.gz | tar -xf -
```

The archive extracts as `99.zarr`. Arrange it under the GPN-Star input contract,
where the directory name records the total count including the target species:

```text
/path/to/multiz100way/
└── 100/
    └── all.zarr/  # the extracted 99.zarr store
```

For example, move or symlink the extracted store to
`/path/to/multiz100way/100/all.zarr`, then pass
`--msa-path /path/to/multiz100way`. A direct
`--msa-path /path/to/multiz100way/100` is also accepted. The logical `100`
directory name is significant even when `all.zarr` is a symlink.

## Inspect a local interval

Querying a short interval is a useful layout check before launching inference:

```python
from gpn.star.data import GenomeMSA

alignment = GenomeMSA(
    "/path/to/multiz100way/100/all.zarr",
    n_species=100,
    in_memory=False,
)
interval = alignment.get_msa(
    "6",
    31_575_665,
    31_575_793,
    strand="+",
    tokenize=False,
)
print(interval.shape)
print(interval[:, 0])
```

Chromosome names must match the store. This example uses the chromosome spelling
from the published archive; do not silently add or strip a `chr` prefix.

Other GPN-Star checkpoints can require different public or local alignment stores.
Consult the checkpoint card and the {doc}`models` compatibility table before
substituting one alignment for another.
