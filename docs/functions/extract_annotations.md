---
title: extract_annotations
parent: Package Functions
nav_order: 13
---

# `extract_annotations`

Extract bounding box annotations from a 3D label volume and save them as CSV files with flexible output formats and coordinate options.

```python
extract_annotations(
    nii_path: str,
    output_path: str,
    view: str = "axial",
    saving_mode: str = "slice",
    data_mode: str = "center",
    target_size: Optional[Tuple[int, int]] = None,
    debug: bool = False
) -> None
```

## Overview

This function converts 3D annotation volumes into structured CSV files containing bounding box information. It identifies each unique annotation label and extracts its spatial extent, offering flexibility in:

- **Anatomical view**: Extract along axial, coronal, or sagittal planes
- **Granularity**: Per-slice or per-volume extraction
- **Data representation**: Centers, full boxes, or center-with-radius format
- **Coordinate adjustment**: Optional padding compensation for alignment with extracted images

This function is designed to work with `extract_slices` to create perfectly aligned image-annotation pairs for machine learning applications.

## Parameters

| Name          | Type                        | Default    | Description                                                                                          |
|---------------|-----------------------------|------------|------------------------------------------------------------------------------------------------------|
| `nii_path`    | `str`                       | *required* | Path to the input annotation volume in `.nii.gz` format.                                            |
| `output_path` | `str`                       | *required* | Directory where CSV files will be saved. Created automatically if it doesn't exist.                 |
| `view`        | `str`                       | `"axial"`  | Anatomical view for extraction: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `saving_mode` | `str`                       | `"slice"`  | Granularity: `"slice"` (CSV per slice) or `"volume"` (single CSV for entire volume).               |
| `data_mode`   | `str`                       | `"center"` | Output format: `"center"`, `"box"`, or `"radius"`.                                                  |
| `target_size` | `Optional[Tuple[int, int]]` | `None`     | Target dimensions (height, width) for coordinate adjustment to account for padding.                |
| `debug`       | `bool`                      | `False`    | If `True`, prints extraction statistics and file paths.                                             |

## Returns

`None` – The function saves CSV files to disk.

## Output Filenames

### Slice Mode (`saving_mode="slice"`)
One CSV per slice containing annotations:
```
<PREFIX>_<VIEW>_<SLICE_NUMBER>.csv
```
**Example**: `patient_042_axial_015.csv`

### Volume Mode (`saving_mode="volume"`)
Single CSV containing all annotations:
```
<PREFIX>.csv
```
**Example**: `patient_042.csv`

## Data Formats

### Center Mode (`data_mode="center"`)
Saves the center coordinates of each annotation's bounding box.

**Volume mode columns**:
| Column     | Description                           |
|------------|---------------------------------------|
| `CENTER_X` | X coordinate of bounding box center   |
| `CENTER_Y` | Y coordinate of bounding box center   |
| `CENTER_Z` | Z coordinate (slice index)            |

**Slice mode columns**:
| Column     | Description                           |
|------------|---------------------------------------|
| `CENTER_X` | X coordinate of bounding box center   |
| `CENTER_Y` | Y coordinate of bounding box center   |

### Box Mode (`data_mode="box"`)
Saves complete bounding box coordinates.

**Volume mode columns**:
| Column  | Description                  |
|---------|------------------------------|
| `X_MIN` | Minimum X coordinate         |
| `Y_MIN` | Minimum Y coordinate         |
| `Z_MIN` | Minimum Z coordinate (slice) |
| `X_MAX` | Maximum X coordinate         |
| `Y_MAX` | Maximum Y coordinate         |
| `Z_MAX` | Maximum Z coordinate (slice) |

**Slice mode columns**:
| Column  | Description          |
|---------|----------------------|
| `X_MIN` | Minimum X coordinate |
| `Y_MIN` | Minimum Y coordinate |
| `X_MAX` | Maximum X coordinate |
| `Y_MAX` | Maximum Y coordinate |

### Radius Mode (`data_mode="radius"`)
Saves center coordinates and radii from center to bounding box edges.

**Volume mode columns**:
| Column     | Description                     |
|------------|---------------------------------|
| `CENTER_X` | X coordinate of center          |
| `CENTER_Y` | Y coordinate of center          |
| `CENTER_Z` | Z coordinate of center          |
| `RADIUS_X` | Radius in X direction           |
| `RADIUS_Y` | Radius in Y direction           |
| `RADIUS_Z` | Radius in Z direction           |

**Slice mode columns**:
| Column     | Description           |
|------------|-----------------------|
| `CENTER_X` | X coordinate of center|
| `CENTER_Y` | Y coordinate of center|
| `RADIUS_X` | Radius in X direction |
| `RADIUS_Y` | Radius in Y direction |

## Anatomical Views

The `view` parameter determines the extraction plane:

| View         | Extraction Axis | Description                    |
|--------------|-----------------|--------------------------------|
| `"axial"`    | Z-axis          | Horizontal slices (top-down)   |
| `"coronal"`  | Y-axis          | Frontal slices (front-back)    |
| `"sagittal"` | X-axis          | Lateral slices (left-right)    |

## Target Size and Coordinate Adjustment

When `target_size` is specified, coordinates are adjusted to account for padding that would be applied to match the target dimensions. This ensures annotations align perfectly with padded images from `extract_slices`.

**Important**: 
- Use the same `target_size` for both `extract_slices` and `extract_annotations`
- Only applied in slice mode (`saving_mode="slice"`)
- Target size must be equal to or larger than the slice dimensions

**Example**:
```python
# Extract images with padding to 512x512
extract_slices(..., target_size=(512, 512))

# Extract annotations with matching adjustment
extract_annotations(..., target_size=(512, 512))
```

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The input file does not exist                                     |
| `ValueError`         | File is not in `.nii.gz` format                                   |
| `ValueError`         | File is not 3D or has invalid dimensions                          |
| `ValueError`         | Invalid `view`, `saving_mode`, or `data_mode`                     |
| `ValueError`         | Target size is smaller than slice dimensions                      |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are accepted
- **3D Volumes Required**: Input must be a 3D NIfTI image
- **Label Detection**: Each unique non-zero value is treated as a separate annotation
- **Coordinate System**: Coordinates are in voxel space (0-indexed)
- **Output Directory**: Automatically created if it doesn't exist
- **Progress Display**: Shows progress bars for label and slice processing

## Examples

### Basic Usage - Center Coordinates
Extract center points for each annotation per slice:

```python
from nidataset.slices import extract_annotations

extract_annotations(
    nii_path="annotations/lesion_mask.nii.gz",
    output_path="labels/centers/",
    view="axial",
    saving_mode="slice",
    data_mode="center"
)
# Creates: labels/centers/lesion_mask_axial_000.csv, ...
```

### Full Bounding Boxes
Extract complete bounding box coordinates:

```python
extract_annotations(
    nii_path="masks/tumor_labels.nii.gz",
    output_path="labels/boxes/",
    view="coronal",
    saving_mode="slice",
    data_mode="box",
    debug=True
)
# Prints: Total slices with annotations extracted: 45
```

### Volume-Based Extraction
Create single CSV with all annotations:

```python
extract_annotations(
    nii_path="segmentations/organs.nii.gz",
    output_path="volume_labels/",
    view="sagittal",
    saving_mode="volume",
    data_mode="box",
    debug=True
)
# Creates: volume_labels/organs.csv (single file)
```

### With Padding Adjustment
Align annotations with padded images:

```python
extract_annotations(
    nii_path="annotations/detection_labels.nii.gz",
    output_path="aligned_labels/",
    view="axial",
    saving_mode="slice",
    data_mode="box",
    target_size=(512, 512),
    debug=True
)
# Coordinates adjusted for 512x512 padded slices
# Prints: Padding adjustment applied: target size (512, 512)
```

### Radius Mode for Circular Representations
Extract center and radius for each annotation:

```python
extract_annotations(
    nii_path="masks/nodules.nii.gz",
    output_path="nodule_params/",
    view="axial",
    saving_mode="volume",
    data_mode="radius",
    debug=True
)
# CSV contains center coordinates and radii in all directions
```

### Complete Image-Annotation Pipeline
Extract aligned images and annotations:

```python
from nidataset.slices import extract_slices, extract_annotations

scan_file = "scans/patient_001.nii.gz"
mask_file = "masks/patient_001_mask.nii.gz"
image_output = "dataset/images/"
label_output = "dataset/labels/"

# Extract images with padding
extract_slices(
    nii_path=scan_file,
    output_path=image_output,
    view="axial",
    target_size=(512, 512),
    normalization="min-max",
    debug=True
)

# Extract annotations with matching adjustment
extract_annotations(
    nii_path=mask_file,
    output_path=label_output,
    view="axial",
    saving_mode="slice",
    data_mode="box",
    target_size=(512, 512),  # Must match image extraction
    debug=True
)

# Result: Perfectly aligned image-label pairs
```

### Multi-View Extraction
Extract annotations from all anatomical views:

```python
from nidataset.slices import extract_annotations

mask_file = "annotations/multi_view_mask.nii.gz"
views = ["axial", "coronal", "sagittal"]

for view in views:
    extract_annotations(
        nii_path=mask_file,
        output_path=f"labels/{view}/",
        view=view,
        saving_mode="slice",
        data_mode="center",
        debug=True
    )
```

### Verifying Annotation Extraction
Check extracted annotations for quality control:

```python
import pandas as pd
from nidataset.slices import extract_annotations

# Extract annotations
extract_annotations(
    nii_path="masks/test_mask.nii.gz",
    output_path="qa/labels/",
    view="axial",
    saving_mode="slice",
    data_mode="box",
    debug=True
)

# Load and verify a sample slice
sample_csv = "qa/labels/test_mask_axial_025.csv"
df = pd.read_csv(sample_csv)

print(f"Slice 25 contains {len(df)} annotations")
print("\nBounding box dimensions:")
df['width'] = df['X_MAX'] - df['X_MIN']
df['height'] = df['Y_MAX'] - df['Y_MIN']
print(df[['width', 'height']].describe())

# Check for suspicious annotations
small = df[(df['width'] < 3) | (df['height'] < 3)]
if not small.empty:
    print(f"\nWarning: {len(small)} very small annotations detected")
```

### Converting Between Formats
Extract in multiple data modes for different use cases:

```python
from nidataset.slices import extract_annotations

mask_file = "annotations/detections.nii.gz"
output_base = "converted_labels/"

# Centers for point detection tasks
extract_annotations(
    nii_path=mask_file,
    output_path=f"{output_base}/centers/",
    view="axial",
    saving_mode="slice",
    data_mode="center"
)

# Boxes for object detection tasks
extract_annotations(
    nii_path=mask_file,
    output_path=f"{output_base}/boxes/",
    view="axial",
    saving_mode="slice",
    data_mode="box"
)

# Radius for size analysis
extract_annotations(
    nii_path=mask_file,
    output_path=f"{output_base}/radius/",
    view="axial",
    saving_mode="volume",
    data_mode="radius"
)
```

### Batch Processing with Different Settings
Process multiple files with varied configurations:

```python
import os
from nidataset.slices import extract_annotations

mask_folder = "masks/"
output_folder = "processed_labels/"

# Configuration for different mask types
configs = {
    'lesion': {'data_mode': 'center', 'saving_mode': 'slice'},
    'organ': {'data_mode': 'box', 'saving_mode': 'volume'},
    'nodule': {'data_mode': 'radius', 'saving_mode': 'slice'}
}

for mask_type, config in configs.items():
    mask_file = os.path.join(mask_folder, f"{mask_type}_mask.nii.gz")
    if os.path.exists(mask_file):
        extract_annotations(
            nii_path=mask_file,
            output_path=f"{output_folder}/{mask_type}/",
            view="axial",
            target_size=(512, 512),
            debug=True,
            **config
        )
```

### Analyzing Annotation Distribution
Extract and analyze annotation patterns:

```python
import pandas as pd
import os
from nidataset.slices import extract_annotations

# Extract annotations
extract_annotations(
    nii_path="masks/dataset_mask.nii.gz",
    output_path="analysis/labels/",
    view="axial",
    saving_mode="slice",
    data_mode="box",
    debug=True
)

# Analyze annotation distribution across slices
csv_files = [f for f in os.listdir("analysis/labels/") if f.endswith('.csv')]
slice_stats = []

for csv_file in sorted(csv_files):
    df = pd.read_csv(f"analysis/labels/{csv_file}")
    slice_num = int(csv_file.split('_')[-1].replace('.csv', ''))
    
    slice_stats.append({
        'slice': slice_num,
        'num_annotations': len(df),
        'avg_width': (df['X_MAX'] - df['X_MIN']).mean() if len(df) > 0 else 0,
        'avg_height': (df['Y_MAX'] - df['Y_MIN']).mean() if len(df) > 0 else 0
    })

stats_df = pd.DataFrame(slice_stats)
print("\nAnnotation Distribution:")
print(f"Total slices: {len(stats_df)}")
print(f"Slices with annotations: {(stats_df['num_annotations'] > 0).sum()}")
print(f"Average annotations per slice: {stats_df['num_annotations'].mean():.2f}")
print(f"Slice with most annotations: {stats_df.loc[stats_df['num_annotations'].idxmax(), 'slice']}")
```

## Typical Workflow

```python
from nidataset.slices import extract_annotations
import pandas as pd

# 1. Define input and output paths
annotation_file = "data/segmentation_mask.nii.gz"
output_folder = "data/extracted_annotations/"

# 2. Extract annotations with desired format
extract_annotations(
    nii_path=annotation_file,
    output_path=output_folder,
    view="axial",
    saving_mode="slice",
    data_mode="box",
    target_size=(512, 512),
    debug=True
)

# 3. Verify extraction
sample_csv = f"{output_folder}/segmentation_mask_axial_010.csv"
df = pd.read_csv(sample_csv)
print(f"Extracted {len(df)} annotations from slice 10")

# 4. Use annotations for training
# - Load corresponding images from extract_slices
# - Create training pairs
# - Feed to detection/segmentation model
```