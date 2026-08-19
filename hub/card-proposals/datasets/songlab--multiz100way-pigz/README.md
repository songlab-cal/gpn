---
# TODO(maintainer): distinguish upstream alignment terms from this archive's terms.
tags:
- biology
- genomics
- multiple-sequence-alignment
- zarr
---

# multiz100way Zarr archive compressed with pigz

This repository contains `99.zarr.tar.gz`, a pigz-compressed tar archive derived
from the `songlab/multiz100way` resource. It was created with pigz level 1 for fast
decompression.

The compressed file is 42,269,901,437 bytes. Check available download and extracted
disk capacity before retrieving it. The GPN test suite and quick starts do **not**
need this archive; they use a checked-in 3.5 KiB fixture.

```bash
hf download songlab/multiz100way-pigz 99.zarr.tar.gz \
  --repo-type dataset \
  --revision 6a9d42a35e7debbba845979dea6064f14d5cb3f9 \
  --local-dir .

echo '4dad7da04db9c804032c0c4c7bbefea58f694fc911e962d28c8df87f356ce4ad  99.zarr.tar.gz' \
  | sha256sum --check
unpigz --stdout 99.zarr.tar.gz | tar -xf -
```

The previous card incorrectly downloaded from `gonzalobenegas/99.zarr`; the
repository above is the actual source of this archive. TODO(maintainer): document
the expected extracted directory layout, Zarr version, species order, source
alignment release, and upstream terms before publication.
