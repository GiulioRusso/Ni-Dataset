---
title: QCConfig
parent: Quality Control (qc)
nav_order: 6
---

# `QCConfig`

User-overridable thresholds and rules for every QC check. Defaults are generic and
domain-neutral (no CT/MR assumptions).

```python
from nidataset.qc import QCConfig

cfg = QCConfig(expected_orientation="RAS", affine_atol=1e-3)
```

## Fields

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `expected_orientation` | `str \| None` | `None` | Expected `aff2axcodes` orientation (e.g. `"RAS"`). `None` disables the check (orientation is still reported). |
| `affine_atol` | `float` | `1e-4` | Tolerance for affine comparisons (singularity, sform/qform, pair/triple alignment). |
| `isotropy_tol` | `float` | `0.05` | Allowed spacing anisotropy as `max/min - 1`. |
| `spacing_range` | `(float, float)` | `(0.1, 10.0)` | Plausible per-axis voxel size in mm. |
| `intensity_range` | `(float, float) \| None` | `None` | Plausible intensity range; `None` disables. |
| `warn_float64` | `bool` | `True` | Warn when dtype is `float64`. |
| `empty_slice_bg_value` | `float` | `0.0` | Voxels ≤ this are background for the empty-slice test. |
| `empty_slice_min_fg_fraction` | `float` | `1e-4` | Foreground fraction below which a slice is "empty". |
| `max_empty_slice_fraction` | `float` | `0.5` | Warn if more than this fraction of slices on an axis are empty. |
| `allowed_labels` | `Sequence[float] \| None` | `None` | Labels a mask/annotation may contain; `None` = binary `{0, 1}`. |
| `max_annotation_outside_fraction` | `float` | `0.0` | Error if more than this fraction of annotation voxels lies outside the mask. |
| `min_component_size` | `int` | `1` | Components smaller than this (voxels) are flagged. |
| `max_components` | `int` | `50` | Warn above this many connected components. |
| `warn_bbox_touches_border` | `bool` | `True` | Warn if the annotation bbox touches the volume border. |
| `max_workers` | `int` | `4` | Thread workers for dataset scans (deterministic output). |

The reasoning behind each default is documented in
[`QC_DESIGN.md`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/QC_DESIGN.md).

## Loading from a file

```python
QCConfig.load("qc.json")    # stdlib, no extra dependency
QCConfig.load("qc.yaml")    # requires the optional `pyyaml` extra
```

- `.json` works out of the box.
- `.yaml` / `.yml` need `pip install nidataset[yaml]`; a clear `ImportError` is
  raised if `pyyaml` is missing.
- Unknown keys raise `ValueError` (typo protection).

See the commented [`qc.example.yaml`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/qc.example.yaml)
and the dependency-free [`qc.example.json`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/qc.example.json).

## Examples

```python
from nidataset.qc import QCConfig, check_dataset

# CT-style: expect RAS, allow slight anisotropy, multi-label annotation
cfg = QCConfig(
    expected_orientation="RAS",
    isotropy_tol=0.10,
    allowed_labels=[0, 1, 2],
    intensity_range=(-1024, 3071),
)
ds = check_dataset("scans/", config=cfg)
```
