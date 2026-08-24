# Published-model scientific fixtures

These small fixtures preserve the behavior of the published models without making
the ordinary test suite download checkpoints or an alignment.

`hg38_chr6_31575665_31575793_multiz100way.npz` contains only the 128 bp interval
used by the archived GPN-MSA notebook and retained GPN-Star quick start. The
100-way slice was read once from an existing SCF Zarr. Stable public provenance is recorded for the
equivalent `songlab/multiz100way-pigz` archive, including its immutable revision,
42.3 GB archive size, and SHA-256; the archive itself was not downloaded. The raw
128 bp slice has its own SHA-256 in the baseline metadata.

The GPN-MSA array was reconstructed with the historical 99- and 89-species lists,
which drops the ten closest primate columns. Tests verify the complete column
relationship and species-list hashes, not merely selected cells. Its displayed
leading and trailing rows and its model output match the GPN-MSA notebook
preserved at the archive snapshot. No full MSA is vendored or downloaded by the
tests.

To regenerate the tiny file from any already-present equivalent 100-way Zarr,
pass its local path explicitly. The utility checks the raw slice and final file
hashes and never downloads or builds an MSA:

```bash
uv run --extra inference python tests/fixtures/regenerate_alignment_fixture.py \
  /path/to/100-way.zarr \
  --output /tmp/hg38_chr6_31575665_31575793_multiz100way.npz
```

`published_model_baseline.json` records immutable Hugging Face revisions, inputs,
floating-point outputs, nucleotide order, and the environment used for the
one-time validation. The model revisions are:

| Model | Repository | Revision |
| --- | --- | --- |
| GPN | `songlab/gpn-brassicales` | `eb9c35d0d18571abe84390d22e74f2b21d319ce3` |
| GPN-MSA | `songlab/gpn-msa-sapiens` | `4a7d4f75449cb2abd560b2af024d76f99233c6db` |
| GPN-Star V100 | `songlab/gpn-star-hg38-v100-200m` | `0c949f132d35619a3eb188b402848c998a3313ae` |
| PhyloGPN | `songlab/PhyloGPN` | `3556db4c469e67d25f0f7a0a6653b48be3eebf51` |
| Sorghum expression | `songlab/gpn-brassicales-gxa-sorghum-v1` | `53209151b497d4840d50526d44c0460b6e6768b7` |

The GPN and GPN-Star values reproduce their retained quick-start notebooks; the
GPN-MSA value reproduces its archived notebook. GPN-Star calibration is bound to
the pinned calibration-table hash.
The Sorghum label order and example row are bound to the pinned dataset's small
`labels.txt` and `test.parquet` artifacts. The PhyloGPN quick-start output is stale
relative to the current published weights: its C-to-T value at position one is
`-0.2926`, while the pinned published revision produces `0.7954897`. The current
value was reproduced exactly with both the pinned Hugging Face remote
implementation on Transformers 4.48.3 and the maintained local AutoClass
implementation on Transformers 5.15.0. The quick start should display the current
result when it is refreshed.

Ordinary tests validate the fixture integrity and likelihood calculations entirely
offline. To deliberately recheck the pinned checkpoints against these fixtures,
run the opt-in suite in an environment where model downloads are acceptable:

```bash
uv run --extra inference pytest -m published_models --run-published-models
```

That command downloads model files, but it does not access an external MSA.
