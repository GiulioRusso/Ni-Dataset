---
title: extract_annotations_dataset
parent: Package Functions
nav_order: 14
---

# `extract_annotations_dataset`

Batch process annotation volumes to extract bounding box coordinates or centers as CSV files, with flexible organization and optional statistics tracking.

```python
extract_annotations_dataset(
    nii_folder: str,
    output_path: str,
    view: str = "axial",
    saving_mode: str = "case",
    extraction_mode: str = "slice",
    data_mode: str = "center",
    target_size: Optional[Tuple[int, int]] = None,
    save_stats: bool = False
) -> None
```

## Overview

This function processes all annotation masks in a dataset folder and extracts annotation coordinates as CSV files. It provides flexible control over:

- **Anatomical view**: Extract from axial, coronal, or sagittal slices
- **Organization**: Group by case or by view
- **Granularity**: Per-slice or per-volume extraction
- **Data format**: Bounding boxes or center points
- **Coordinate adjustment**: Optional padding compensation for alignment with extracted images

The function is designed to work in tandem with `extract_slices_dataset` to create aligned image-annotation pairs for machine learning tasks.

## Parameters

| Name              | Type                        | Default    | Description                                                                                          |
|-------------------|-----------------------------|------------|------------------------------------------------------------------------------------------------------|
| `nii_folder`      | `str`                       | *required* | Path to the directory containing annotation volumes in `.nii.gz` format.                            |
| `output_path`     | `str`                       | *required* | Root directory where extracted annotations will be saved.                                           |
| `view`            | `str`                       | `"axial"`  | Anatomical view for extraction: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `saving_mode`     | `str`                       | `"case"`   | Organization mode: `"case"` (folder per file) or `"view"` (shared folder).                         |
| `extraction_mode` | `str`                       | `"slice"`  | Granularity: `"slice"` (CSV per slice) or `"volume"` (single CSV per case).                        |
| `data_mode`       | `str`                       | `"center"` | Output format: `"center"` (point coordinates) or `"box"` (bounding box coordinates).               |
| `target_size`     | `Optional[Tuple[int, int]]` | `None`     | Target dimensions (height, width) for coordinate adjustment to account for padding.                |
| `save_stats`      | `bool`                      | `False`    | If `True`, saves annotation statistics as `<view>_annotations_stats.csv`.                          |

## Returns

`None` – The function saves CSV files to disk.

## Output Organization

### Saving Modes

#### Case Mode (`saving_mode="case"`)
Creates a separate folder for each annotation file:
```
output_path/
├── case_001/
│   └── axial/
│       ├── case_001_axial_0.csv
│       ├── case_001_axial_1.csv
│       └── ...
├── case_002/
│   └── axial/
│       └── ...
```

#### View Mode (`saving_mode="view"`)
Groups all annotations in a single view folder:
```
output_path/
└── axial/
    ├── case_001_axial_0.csv
    ├── case_001_axial_1.csv
    ├── case_002_axial_0.csv
    └── ...
```

### Extraction Modes

#### Slice Mode (`extraction_mode="slice"`)
Creates one CSV per slice:
- Filename pattern: `<PREFIX>_<VIEW>_<SLICE_NUMBER>.csv`
- Example: `patient_042_axial_15.csv`

#### Volume Mode (`extraction_mode="volume"`)
Creates one CSV for the entire volume:
- Filename pattern: `<PREFIX>.csv`
- Contains annotations from all slices with slice index information

## Data Formats

### Center Mode (`data_mode="center"`)
CSV contains the center coordinates of each annotation:

| Column | Description                           |
|--------|---------------------------------------|
| `X`    | X coordinate of annotation center     |
| `Y`    | Y coordinate of annotation center     |
| `Z`    | Z coordinate (slice index)            |

### Box Mode (`data_mode="box"`)
CSV contains full bounding box coordinates:

| Column  | Description                                  |
|---------|----------------------------------------------|
| `X_MIN` | Minimum X coordinate                         |
| `Y_MIN` | Minimum Y coordinate                         |
| `Z_MIN` | Minimum Z coordinate (slice index)           |
| `X_MAX` | Maximum X coordinate                         |
| `Y_MAX` | Maximum Y coordinate                         |
| `Z_MAX` | Maximum Z coordinate (slice index)           |

## Anatomical Views

The `view` parameter determines which axis to extract along:

| View         | Extraction Axis | Description                    |
|--------------|-----------------|--------------------------------|
| `"axial"`    | Z-axis          | Horizontal slices (top-down)   |
| `"coronal"`  | Y-axis          | Frontal slices (front-back)    |
| `"sagittal"` | X-axis          | Lateral slices (left-right)    |

## Target Size and Coordinate Adjustment

When `target_size` is specified, coordinates are adjusted to account for padding applied during image extraction. This ensures alignment between images and annotations.

**Important**: Use the same `target_size` value for both `extract_slices_dataset` and `extract_annotations_dataset`.

**Example**:
```python
# Extract images with padding to 512x512
extract_slices_dataset(..., target_size=(512, 512))

# Extract annotations with matching adjustment
extract_annotations_dataset(..., target_size=(512, 512))
```

## Statistics File

When `save_stats=True`, a CSV file is created with annotation counts:

| Column              | Description                        |
|---------------------|------------------------------------|
| `FILENAME`          | Annotation file name               |
| `NUM_ANNOTATIONS`   | Number of annotations in the file  |
| `TOTAL_ANNOTATIONS` | Sum across all files (last row)    |

The file is named `<view>_annotations_stats.csv` and saved in `output_path`.

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |
| `ValueError`         | Invalid `view`, `saving_mode`, `extraction_mode`, or `data_mode`  |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **Progress Display**: Shows progress bar with current file being processed
- **Error Handling**: Files that fail are skipped with error messages
- **Coordinate System**: Coordinates are in voxel space (0-indexed)
- **Multiple Annotations**: Each connected component gets its own row in the CSV

## Examples

### Basic Usage - Slice-Based Extraction
Extract center coordinates for each axial slice:

```python
from nidataset.slices import extract_annotations_dataset

extract_annotations_dataset(
    nii_folder="dataset/annotations/",
    output_path="extracted/annotations/",
    view="axial",
    saving_mode="case",
    extraction_mode="slice",
    data_mode="center"
)
# Creates: extracted/annotations/case_001/axial/case_001_axial_0.csv, ...
```

### With Statistics Tracking
Enable annotation statistics for dataset overview:

```python
extract_annotations_dataset(
    nii_folder="dataset/masks/",
    output_path="output/labels/",
    view="coronal",
    saving_mode="view",
    extraction_mode="slice",
    data_mode="center",
    save_stats=True
)
# Creates: output/labels/coronal_annotations_stats.csv
```

### Full Bounding Boxes with Padding Adjustment
Extract complete bounding boxes with coordinate adjustment:

```python
extract_annotations_dataset(
    nii_folder="data/segmentations/",
    output_path="data/bbox_labels/",
    view="axial",
    saving_mode="case",
    extraction_mode="slice",
    data_mode="box",
    target_size=(512, 512),
    save_stats=True
)
# Coordinates adjusted for 512x512 padded images
```

### Volume-Based Extraction
Create single CSV per case with all annotations:

```python
extract_annotations_dataset(
    nii_folder="annotations/",
    output_path="volume_labels/",
    view="sagittal",
    saving_mode="case",
    extraction_mode="volume",
    data_mode="center",
    save_stats=True
)
# Creates: volume_labels/case_001/sagittal/case_001.csv
```

### Complete Image-Annotation Pipeline
Extract aligned images and annotations for training:

```python
from nidataset.slices import extract_slices_dataset, extract_annotations_dataset

# 1. Extract images with padding
extract_slices_dataset(
    nii_folder="data/scans/",
    output_path="training_data/images/",
    view="axial",
    saving_mode="case",
    target_size=(512, 512),
    normalization="min-max",
    save_stats=True
)

# 2. Extract annotations with matching adjustment
extract_annotations_dataset(
    nii_folder="data/masks/",
    output_path="training_data/labels/",
    view="axial",
    saving_mode="case",
    extraction_mode="slice",
    data_mode="box",
    target_size=(512, 512),  # Must match image extraction
    save_stats=True
)

# Result: Aligned image-annotation pairs ready for training
```

### Multi-View Extraction
Extract annotations from all three anatomical views:

```python
from nidataset.slices import extract_annotations_dataset

views = ["axial", "coronal", "sagittal"]
base_path = "multi_view_annotations/"

for view in views:
    print(f"Extracting {view} view...")
    extract_annotations_dataset(
        nii_folder="dataset/labels/",
        output_path=base_path,
        view=view,
        saving_mode="view",
        extraction_mode="slice",
        data_mode="center",
        save_stats=True
    )

# Creates separate folders for each view with statistics
```

### Analyzing Statistics
Review annotation distribution across dataset:

```python
import pandas as pd
from nidataset.slices import extract_annotations_dataset

# Extract with statistics
extract_annotations_dataset(
    nii_folder="annotations/",
    output_path="results/",
    view="axial",
    saving_mode="view",
    extraction_mode="slice",
    data_mode="center",
    save_stats=True
)

# Load and analyze statistics
stats = pd.read_csv("results/axial_annotations_stats.csv")

# Remove total row for per-file analysis
per_file = stats[stats['FILENAME'] != 'TOTAL_ANNOTATIONS'].copy()
per_file['NUM_ANNOTATIONS'] = pd.to_numeric(per_file['NUM_ANNOTATIONS'])

print("Annotation Statistics:")
print(f"  Total files: {len(per_file)}")
print(f"  Files with annotations: {(per_file['NUM_ANNOTATIONS'] > 0).sum()}")
print(f"  Average annotations per file: {per_file['NUM_ANNOTATIONS'].mean():.2f}")
print(f"  Max annotations: {per_file['NUM_ANNOTATIONS'].max()}")
print(f"  Min annotations: {per_file['NUM_ANNOTATIONS'].min()}")

# Files without annotations
empty = per_file[per_file['NUM_ANNOTATIONS'] == 0]
if not empty.empty:
    print(f"\nWarning: {len(empty)} files have no annotations:")
    print(empty['FILENAME'].tolist())
```

### Quality Control Workflow
Verify annotation extraction quality:

```python
import pandas as pd
from nidataset.slices import extract_annotations_dataset

# Extract annotations
extract_annotations_dataset(
    nii_folder="masks/",
    output_path="qa/annotations/",
    view="axial",
    saving_mode="case",
    extraction_mode="slice",
    data_mode="box",
    save_stats=True
)

# Check a sample annotation file
sample_csv = "qa/annotations/case_001/axial/case_001_axial_10.csv"
df = pd.read_csv(sample_csv)

print(f"Sample slice has {len(df)} annotations")
print("\nBounding box sizes:")
df['width'] = df['X_MAX'] - df['X_MIN']
df['height'] = df['Y_MAX'] - df['Y_MIN']
print(df[['width', 'height']].describe())

# Identify potential issues
small_boxes = df[(df['width'] < 5) | (df['height'] < 5)]
if not small_boxes.empty:
    print(f"\nWarning: {len(small_boxes)} very small annotations detected")
```

## Typical Workflow

```python
from nidataset.slices import extract_annotations_dataset
import pandas as pd

# 1. Define paths
annotation_folder = "data/segmentation_masks/"
output_folder = "data/extracted_labels/"

# 2. Extract annotations with statistics
extract_annotations_dataset(
    nii_folder=annotation_folder,
    output_path=output_folder,
    view="axial",
    saving_mode="case",
    extraction_mode="slice",
    data_mode="box",
    target_size=(512, 512),
    save_stats=True
)

# 3. Review statistics
stats = pd.read_csv(f"{output_folder}/axial_annotations_stats.csv")
print(stats.head())

# 4. Use extracted annotations for training
# - Load corresponding images from extract_slices_dataset
# - Create dataloaders with image-annotation pairs
# - Train detection or segmentation models
```