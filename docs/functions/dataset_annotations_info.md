---
title: dataset_annotations_info
parent: Package Functions
nav_order: 3
---

# `dataset_annotations_info`

Extract 3D bounding boxes for all connected regions of a specific label value from annotation volumes and save them as a CSV file.

```python
dataset_annotations_info(
    nii_folder: str,
    output_path: str,
    annotation_value: int = 1
) -> None
```

## Overview

This function processes all annotation masks in a folder to identify and extract bounding boxes around labeled regions. It uses connected component analysis to detect separate instances of the same label value, making it useful for:

- Multi-instance object detection datasets
- Lesion or tumor localization in medical images
- Organ or anatomical structure boundary extraction
- Quality control and annotation verification

Each connected region with the specified annotation value gets its own bounding box, and all boxes are saved to `dataset_annotations_info.csv`.

## Parameters

| Name               | Type  | Default    | Description                                                                                  |
|--------------------|-------|------------|----------------------------------------------------------------------------------------------|
| `nii_folder`       | `str` | *required* | Path to the directory containing annotation volumes in `.nii.gz` format.                    |
| `output_path`      | `str` | *required* | Directory where the CSV file will be saved. Created automatically if it doesn't exist.      |
| `annotation_value` | `int` | `1`        | Voxel value representing the region of interest in the annotation masks.                    |

## Returns

`None` – The function saves results to disk.

## Output File

### CSV Structure
The function creates `dataset_annotations_info.csv` in the specified output directory with two columns:

| Column      | Description                                                                           |
|-------------|---------------------------------------------------------------------------------------|
| `FILENAME`  | Name of the annotation file                                                          |
| `3D_BOXES`  | List of bounding boxes, each as `[xmin, ymin, zmin, xmax, ymax, zmax]`              |

### Bounding Box Format
Each bounding box is a list of 6 integers representing the minimum and maximum coordinates:
```
[X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX]
```

**Example CSV content**:
```
FILENAME,3D_BOXES
case_001_mask.nii.gz,"[[45, 67, 23, 89, 112, 56], [120, 130, 40, 145, 155, 65]]"
case_002_mask.nii.gz,"[[50, 70, 30, 95, 115, 60]]"
```

## Connected Component Analysis

The function uses connected component labeling to identify separate instances of the annotation value. This means:

- **Multiple regions**: If the same label value appears in disconnected regions, each region gets its own bounding box
- **Single region**: If all voxels with the annotation value are connected, only one bounding box is created
- **Empty masks**: Files with no voxels matching the annotation value will have an empty list of boxes

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **Progress Display**: Shows a progress bar with file count during processing
- **Error Handling**: Files that fail to process are skipped with error messages
- **Output Directory**: Automatically created if it doesn't exist
- **Coordinate System**: Bounding boxes use voxel coordinates (0-indexed)

## Examples

### Basic Usage
Extract bounding boxes for annotation value 1:

```python
from nidataset.utility import dataset_annotations_info

dataset_annotations_info(
    nii_folder="dataset/annotations/",
    output_path="results/bboxes/",
    annotation_value=1
)
# Creates: results/bboxes/dataset_annotations_info.csv
```

### Multiple Annotation Values
Process different anatomical structures separately:

```python
# Extract liver annotations (value = 1)
dataset_annotations_info(
    nii_folder="dataset/masks/",
    output_path="results/liver_boxes/",
    annotation_value=1
)

# Extract kidney annotations (value = 2)
dataset_annotations_info(
    nii_folder="dataset/masks/",
    output_path="results/kidney_boxes/",
    annotation_value=2
)
```

### Analyzing Results
Load and analyze the extracted bounding boxes:

```python
import pandas as pd
from nidataset.utility import dataset_annotations_info

# Extract bounding boxes
dataset_annotations_info(
    nii_folder="annotations/",
    output_path="output/",
    annotation_value=1
)

# Load and analyze
df = pd.read_csv("output/dataset_annotations_info.csv")
print(f"Total files processed: {len(df)}")
print(f"Files with annotations: {df['3D_BOXES'].apply(lambda x: len(eval(x)) > 0).sum()}")

# Check a specific file
import ast
boxes = ast.literal_eval(df.loc[0, '3D_BOXES'])
print(f"Number of regions in first file: {len(boxes)}")
for i, box in enumerate(boxes):
    xmin, ymin, zmin, xmax, ymax, zmax = box
    print(f"Region {i+1}: Size = {xmax-xmin}×{ymax-ymin}×{zmax-zmin}")
```

### Verifying Annotations
Use bounding boxes to verify annotation quality:

```python
import nibabel as nib
import ast
import pandas as pd
from nidataset.utility import dataset_annotations_info

# Extract boxes
dataset_annotations_info(
    nii_folder="masks/",
    output_path="output/",
    annotation_value=1
)

# Check for suspicious small or large boxes
df = pd.read_csv("output/dataset_annotations_info.csv")

for idx, row in df.iterrows():
    boxes = ast.literal_eval(row['3D_BOXES'])
    
    for box in boxes:
        xmin, ymin, zmin, xmax, ymax, zmax = box
        volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        
        if volume < 10:
            print(f"Warning: Very small region in {row['FILENAME']}: volume={volume}")
        elif volume > 100000:
            print(f"Warning: Very large region in {row['FILENAME']}: volume={volume}")
```

### Complete Workflow
Extract boxes and create visualization metadata:

```python
import pandas as pd
import ast
from nidataset.utility import dataset_annotations_info

# 1. Extract all bounding boxes
dataset_annotations_info(
    nii_folder="dataset/segmentations/",
    output_path="dataset/metadata/",
    annotation_value=1
)

# 2. Load results
df = pd.read_csv("dataset/metadata/dataset_annotations_info.csv")

# 3. Create summary statistics
summary = []
for idx, row in df.iterrows():
    boxes = ast.literal_eval(row['3D_BOXES'])
    summary.append({
        'filename': row['FILENAME'],
        'num_regions': len(boxes),
        'has_annotations': len(boxes) > 0
    })

summary_df = pd.DataFrame(summary)
print(f"\nDataset Summary:")
print(f"Total files: {len(summary_df)}")
print(f"Files with annotations: {summary_df['has_annotations'].sum()}")
print(f"Average regions per file: {summary_df['num_regions'].mean():.2f}")
```

## Typical Workflow

```python
from nidataset.utility import dataset_annotations_info

# 1. Prepare annotation folder
annotation_folder = "data/segmentation_masks/"
output_folder = "data/bounding_boxes/"

# 2. Extract bounding boxes for target structure
dataset_annotations_info(
    nii_folder=annotation_folder,
    output_path=output_folder,
    annotation_value=1
)

# 3. Review the output
import pandas as pd
df = pd.read_csv("data/bounding_boxes/dataset_annotations_info.csv")
print(df.head())

# 4. Use boxes for downstream tasks
# - Object detection training
# - Region cropping
# - Statistical analysis
```