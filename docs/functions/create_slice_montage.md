---
title: create_slice_montage
parent: Package Functions
nav_order: 49
---

# `create_slice_montage`

Create a montage (grid) of evenly-spaced slices from a NIfTI volume for quick visual inspection.

```python
create_slice_montage(
    nii_path: str,
    output_path: str,
    view: str = "axial",
    num_slices: int = 16,
    cols: int = 4,
    debug: bool = False
) -> str
```

## Overview

This function extracts evenly-spaced slices from a 3D NIfTI volume and arranges them in a grid layout, saving the result as a single PNG image. This provides a compact overview of the entire volume in one image.

This is useful for:
- Quick visual inspection of volumes without a NIfTI viewer
- Generating overview thumbnails for large datasets
- Creating figures for reports and documentation
- Comparing volumes at a glance

The output is saved as `<PREFIX>_montage_<VIEW>.png`.

## Parameters

| Name         | Type   | Default    | Description                                                                          |
|--------------|--------|------------|--------------------------------------------------------------------------------------|
| `nii_path`   | `str`  | *required* | Path to the input `.nii.gz` file.                                                   |
| `output_path`| `str`  | *required* | Directory where the montage image will be saved.                                    |
| `view`       | `str`  | `"axial"`  | Anatomical view: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `num_slices` | `int`  | `16`       | Number of slices to include in the montage.                                         |
| `cols`       | `int`  | `4`        | Number of columns in the grid layout.                                               |
| `debug`      | `bool` | `False`    | If `True`, logs the output path and grid dimensions.                                |

## Returns

`str` – Path to the saved montage image.

## Output File

The montage image is saved as:
```
<PREFIX>_montage_<VIEW>.png
```

**Example**: Input `scan_001.nii.gz` → Output `scan_001_montage_axial.png`

### Grid Layout
The grid is organized as:
- **Columns**: Defined by the `cols` parameter
- **Rows**: Automatically calculated as `ceil(num_slices / cols)`
- **Slice spacing**: Evenly distributed across the full range of the volume

**Example** with `num_slices=16` and `cols=4`:
```
┌──────┬──────┬──────┬──────┐
│ s001 │ s020 │ s040 │ s060 │
├──────┼──────┼──────┼──────┤
│ s080 │ s100 │ s120 │ s140 │
├──────┼──────┼──────┼──────┤
│ s160 │ s180 │ s200 │ s220 │
├──────┼──────┼──────┼──────┤
│ s240 │ s260 │ s280 │ s300 │
└──────┴──────┴──────┴──────┘
```

## Exceptions

| Exception           | Condition                              |
|---------------------|----------------------------------------|
| `FileNotFoundError` | Input file does not exist              |
| `ValueError`        | Invalid view parameter                 |
| `ValueError`        | File is not a valid `.nii.gz` format   |

## Usage Notes

- **Intensity Normalization**: Each slice is independently normalized to [0, 255] for display
- **Grayscale Output**: The montage is saved as a grayscale image
- **Empty Cells**: If `num_slices` doesn't fill the grid evenly, remaining cells are black

## Examples

### Basic Usage
Create a 4x4 montage of axial slices:

```python
from nidataset.visualization import create_slice_montage

create_slice_montage(
    nii_path="brain_scan.nii.gz",
    output_path="thumbnails/"
)
# Output: thumbnails/brain_scan_montage_axial.png
```

### Custom Grid Layout
Create a wider montage with more slices:

```python
create_slice_montage(
    nii_path="scan.nii.gz",
    output_path="montages/",
    view="coronal",
    num_slices=24,
    cols=6
)
# Creates a 4×6 grid of coronal slices
```

### All Three Views

```python
from nidataset.visualization import create_slice_montage

for view in ["axial", "coronal", "sagittal"]:
    create_slice_montage(
        nii_path="scan.nii.gz",
        output_path="montages/",
        view=view,
        num_slices=16,
        cols=4
    )
```

### Dataset Thumbnails
Generate montages for every scan in a folder:

```python
import os
from nidataset.visualization import create_slice_montage

input_folder = "dataset/scans/"
for fname in sorted(os.listdir(input_folder)):
    if fname.endswith(".nii.gz"):
        create_slice_montage(
            nii_path=os.path.join(input_folder, fname),
            output_path="dataset/thumbnails/",
            num_slices=16,
            cols=4
        )
```

## Typical Workflow

```python
from nidataset.visualization import create_slice_montage

# 1. Generate a quick overview of a volume
out_path = create_slice_montage(
    nii_path="data/patient_scan.nii.gz",
    output_path="data/previews/",
    view="axial",
    num_slices=20,
    cols=5,
    debug=True
)

# 2. Open the montage for review
print(f"Montage saved: {out_path}")
```
