---
title: check_dataset
parent: Quality Control (qc)
nav_order: 4
---

# `check_dataset`

Run QC over a whole dataset and summarize cross-dataset coherence.

```python
check_dataset(
    path: str,
    config: Optional[QCConfig] = None,
    max_workers: Optional[int] = None,
) -> DatasetReport
```

## Overview

`path` is **auto-detected**:

- a **folder** of NIfTI files → runs [`check_volume`](check_volume.md) on each;
- a **`.csv` manifest** → runs [`check_pair`](check_pair.md) /
  [`check_triple`](check_triple.md) per row.

Each item is checked independently (lazily — one volume in memory at a time), then
the cross-dataset distributions are computed: a dataset where 287 volumes are RAS
and 13 are LAS has a silent orientation bug the per-file view never surfaces.

Reads are lazy and may run on a thread pool (I/O-bound `nib.load`), but the
returned `DatasetReport` is always **deterministic and sorted**.

## CSV manifest format

Columns `image,mask[,annotation]`, with or without a header. Header column names
are matched case-insensitively (`image`/`img`/`ct`/`cta`/`volume`,
`mask`/`brain`, `annotation`/`label`/`lesion`/`bbox`/`gt`); otherwise positional
order is assumed. Rows with a third column are treated as triples, two-column rows
as pairs.

```csv
image,mask,annotation
case01/ct.nii.gz,case01/brain.nii.gz,case01/lesion.nii.gz
case02/ct.nii.gz,case02/brain.nii.gz,case02/lesion.nii.gz
```

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str` | *required* | Folder of NIfTI files, or a `.csv` manifest. |
| `config` | `QCConfig` | `QCConfig()` | Thresholds; see [`QCConfig`](qcconfig.md). |
| `max_workers` | `int` | `config.max_workers` | Thread workers. `1` disables parallelism; output order is deterministic regardless. |

## Returns

`DatasetReport` with:

- `items` — per-item `Report` objects (sorted);
- `status` / `counts()` — worst status and per-item `ok`/`warning`/`error` tally;
- `worst_first()` — items ranked worst-first;
- `distributions` — `orientation`, `spacing`, `shape`, `dtype` counts plus
  `outliers` (paths not sharing the majority value).

## Examples

```python
from nidataset.qc import check_dataset, QCConfig

ds = check_dataset("scans/", QCConfig(expected_orientation="RAS"))

print(ds.counts())                        # {'ok': 280, 'warning': 17, 'error': 3}
print(ds.distributions["orientation"])    # {'RAS': 287, 'LAS': 13}
print(ds.distributions["outliers"]["orientation"])  # the 13 LAS file paths

for item in ds.worst_first()[:5]:         # the 5 worst items
    print(item.status, item.target)
```

```python
# CSV of detection triples
ds = check_dataset("triples.csv")
```
