---
title: check_triple
parent: Quality Control (qc)
nav_order: 3
---

# `check_triple`

Check coherence between an **image**, a **mask** (brain) and an **annotation**
(lesion / bounding box).

```python
check_triple(
    image_path: str,
    mask_path: str,
    annotation_path: str,
    config: Optional[QCConfig] = None,
) -> Report
```

## Overview

Extends [`check_pair`](check_pair.md) to the three-volume case. It runs the
image↔mask checks, the same geometry checks for image↔annotation, then the
annotation-specific semantics: non-emptiness, containment inside the mask,
bounding-box bounds and connected-component analysis. Pair and triple share the
same geometry-match core, so the alignment guarantee is identical.

## Checks performed

| Check name | Status on failure | What it catches |
|------------|-------------------|-----------------|
| `shape_match_mask`, `shape_match_annotation` | error | Differing spatial shapes. |
| `affine_match_mask`, `affine_match_annotation` | error | World-space misalignment (`translation_mm` reported). |
| `labels_mask`, `labels_annotation` | error | Labels outside the allowed set (default binary). |
| `nonempty_mask`, `nonempty_annotation` | error | Zero active voxels (classic empty-bbox bug). |
| `containment` | error | Fraction of annotation voxels outside the mask above `max_annotation_outside_fraction`. |
| `bbox_annotation` | warning | Annotation bounding box touches the volume border (possible clipping). |
| `components_annotation` | warning | Too many components (`max_components`) or sub-`min_component_size` blobs. |

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image_path` | `str` | *required* | Path to the image NIfTI. |
| `mask_path` | `str` | *required* | Path to the mask NIfTI. |
| `annotation_path` | `str` | *required* | Path to the annotation NIfTI. |
| `config` | `QCConfig` | `QCConfig()` | Thresholds; see [`QCConfig`](qcconfig.md). |

## Returns

`Report` (`kind="triple"`). `report.meta` includes `bbox_annotation` and
`components_annotation` (count + largest sizes) alongside the standard alignment
values.

## Examples

```python
from nidataset.qc import check_triple, QCConfig

cfg = QCConfig(expected_orientation="RAS", max_annotation_outside_fraction=0.0)
rep = check_triple("ct.nii.gz", "brain.nii.gz", "lesion.nii.gz", cfg)

print(rep.status)
for r in rep.issues():
    print(r.name, "→", r.message)
```
