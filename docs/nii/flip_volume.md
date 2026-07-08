---
title: flip_volume
parent: Quick Operations (nii)
nav_order: 3
---

# `flip_volume`

Mirror a NIfTI volume along a voxel axis, updating the affine to preserve world space.

```python
flip_volume(
    nii_path: str,
    output_path: str,
    axis: int = 0,
    debug: bool = False
) -> str
```

## Overview

Flipping reverses voxel order along one axis (`numpy.flip`) and updates the affine
so world coordinates are preserved. This matters: a flip that leaves the affine
untouched silently swaps left/right and corrupts orientation metadata.

The output is saved as `<PREFIX>_flip<axis>.nii.gz`.

## Parameters

| Name          | Type   | Default    | Description                                     |
|---------------|--------|------------|-------------------------------------------------|
| `nii_path`    | `str`  | *required* | Path to the input `.nii.gz` file.              |
| `output_path` | `str`  | *required* | Directory where the flipped volume is saved.   |
| `axis`        | `int`  | `0`        | Voxel axis to flip (`0`, `1`, or `2`).         |
| `debug`       | `bool` | `False`    | If `True`, logs the output path.               |

## Returns

`str` – Path to the saved flipped file.

## Output File

```
<PREFIX>_flip<axis>.nii.gz
```

**Example**: Input `scan.nii.gz` with `axis=0` → Output `scan_flip0.nii.gz`

## Exceptions

| Exception           | Condition                                    |
|---------------------|----------------------------------------------|
| `FileNotFoundError` | Input file does not exist                    |
| `ValueError`        | Input file is not a valid `.nii.gz` / `.nii` |
| `ValueError`        | `axis` is not `0`, `1`, or `2`               |

## Usage Notes

- **Lossless**: voxels are reordered, not interpolated — intensities are unchanged.
- **Affine-correct**: applying the same flip twice restores the original volume.

## Examples

### Mirror along the first axis

```python
from nidataset.spatial import flip_volume

flip_volume("scan.nii.gz", "out/", axis=0)
# Output: out/scan_flip0.nii.gz
```

## Typical Workflow

```python
from nidataset.spatial import flip_volume

# Left-right augmentation without breaking orientation metadata
out = flip_volume("data/scan.nii.gz", "data/aug/", axis=0, debug=True)
print(f"Flipped volume: {out}")
```
