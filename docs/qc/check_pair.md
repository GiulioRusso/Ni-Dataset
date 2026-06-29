---
title: check_pair
parent: Quality Control (qc)
nav_order: 2
---

# `check_pair`

Check coherence between an **image** and a **mask** (e.g. brain mask).

```python
check_pair(
    image_path: str,
    mask_path: str,
    config: Optional[QCConfig] = None,
) -> Report
```

## Overview

The single most valuable check for detection/segmentation. Matching *shape* is not
enough: an image and a mask can have identical shapes yet different affines, which
places them at **different world positions** — the model then trains on a mask
silently shifted relative to its image. `check_pair` catches exactly this.

## Checks performed

| Check name | Status on failure | What it catches |
|------------|-------------------|-----------------|
| `shape_match_mask` | error | Image and mask have different spatial shapes. |
| `affine_match_mask` | error | Affines differ beyond `affine_atol`. Reports `max_abs_diff` and the world-space `translation_mm`. |
| `labels_mask` | error | Mask contains labels outside the allowed set (default binary `{0, 1}`). |
| `nonempty_mask` | error | Mask has zero active voxels. |

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image_path` | `str` | *required* | Path to the image NIfTI. |
| `mask_path` | `str` | *required* | Path to the mask NIfTI. |
| `config` | `QCConfig` | `QCConfig()` | Thresholds; see [`QCConfig`](qcconfig.md). |

## Returns

`Report` (`kind="pair"`). The `affine_match_mask` result's `value` is
`{"max_abs_diff": float, "translation_mm": float}` — the misalignment magnitude.

## Examples

```python
from nidataset.qc import check_pair

rep = check_pair("ct.nii.gz", "brain_mask.nii.gz")
print(rep.status)

# Inspect the alignment offset directly
for r in rep.results:
    if r.name == "affine_match_mask":
        print("world shift (mm):", r.value["translation_mm"])
```
