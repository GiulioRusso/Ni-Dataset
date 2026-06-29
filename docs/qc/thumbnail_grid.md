---
title: thumbnail_grid
parent: Quality Control (qc)
nav_order: 5
---

# `thumbnail_grid`

Render a PNG grid of central orthogonal slices for fast visual QC.

```python
thumbnail_grid(
    specs: List[Dict[str, Optional[str]]],
    output_path: str,
    cell_size: int = 160,
) -> str
```

## Overview

One **row per item**, three **columns** (sagittal / coronal / axial central
slices), so hundreds of volumes can be eyeballed at once. For pairs/triples the
mask and annotation are overlaid in translucent colour on the image (mask green,
annotation red), making spatial **misalignment obvious at a glance**.

Intensity is normalised on the 1st/99th **percentiles** (not raw min/max) so a few
extreme voxels don't render every thumbnail black. Uses only Pillow + numpy +
nibabel — no new dependencies.

## Spec format

Each spec is a dict:

| Key | Required | Description |
|-----|----------|-------------|
| `image` | yes | Path to the image NIfTI. |
| `mask` | no | Mask overlaid in green where shape matches the image. |
| `annotation` | no | Annotation overlaid in red where shape matches. |
| `label` | no | Optional caption (defaults to the image filename). |

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `specs` | `List[Dict]` | *required* | One dict per row (see above). |
| `output_path` | `str` | *required* | Output `.png` path (parent dirs created). |
| `cell_size` | `int` | `160` | Pixel size of each square plane cell. |

## Returns

`str` — the written PNG path.

## Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | No specs provided, or none of the images are readable. |

Unreadable individual images are skipped with a logged warning so the grid still
renders.

## Examples

```python
from nidataset.qc import thumbnail_grid

# Pair overlay
thumbnail_grid(
    [{"image": "ct.nii.gz", "mask": "brain.nii.gz", "annotation": "lesion.nii.gz"}],
    "qc/case01.png",
)

# Many volumes at once
specs = [{"image": p} for p in ["a.nii.gz", "b.nii.gz", "c.nii.gz"]]
thumbnail_grid(specs, "qc/overview.png")
```

From the CLI, the same grid is produced with `niqc <path> --thumbnails DIR`.
