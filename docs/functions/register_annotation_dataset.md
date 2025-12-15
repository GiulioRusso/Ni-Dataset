---
title: register_annotation_dataset
parent: Package Functions
nav_order: 27
---

# `register_annotation_dataset`

Apply saved transformations to all annotations in a dataset folder using previously computed registration transforms.

```python
register_annotation_dataset(
    annotation_folder: str,
    transform_folder: str,
    reference_folder: str,
    output_path: str,
    recalculate_bbox: bool = True,
    saving_mode: str = "case",
    debug: bool = False
) -> None
```

## Overview

This function processes all annotation files (typically bounding boxes) in a dataset by applying previously computed transformations from image registration. Each annotation is transformed using its corresponding transformation file and reference registered image, with optional bounding box recalculation for maintaining axis-aligned boxes.

**Batch transformation pipeline**:
1. **Discovery**: Locates all annotations in input folder
2. **Prefix matching**: Finds corresponding transforms and references (handles annotation suffixes)
3. **Transformation**: Applies saved transformations with nearest neighbor interpolation
4. **Recalculation** (optional): Creates new axis-aligned bounding boxes
5. **Organization**: Saves results according to `saving_mode`

This is essential for:
- Batch processing object detection annotations
- Creating normalized bounding box datasets
- Maintaining axis-aligned boxes across registrations
- Preparing training data for detection models
- Building standardized annotation collections for machine learning

## Parameters

| Name                | Type   | Default    | Description                                                                                          |
|---------------------|--------|------------|------------------------------------------------------------------------------------------------------|
| `annotation_folder` | `str`  | *required* | Path to folder containing input annotation `.nii.gz` files (typically bounding boxes).              |
| `transform_folder`  | `str`  | *required* | Path to folder containing transformation files (`.tfm`). Location depends on `saving_mode`.         |
| `reference_folder`  | `str`  | *required* | Path to folder containing registered reference images. Location depends on `saving_mode`.           |
| `output_path`       | `str`  | *required* | Base directory for all outputs. Organization depends on `saving_mode`.                              |
| `recalculate_bbox`  | `bool` | `True`     | If `True`, creates axis-aligned boxes. If `False`, preserves deformed shapes.                      |
| `saving_mode`       | `str`  | `"case"`   | Organization mode: `"case"` (per-case subfolders) or `"folder"` (single directory).               |
| `debug`             | `bool` | `False`    | If `True`, prints detailed information and warnings for each annotation.                            |

## Saving Mode Behavior

The `saving_mode` parameter controls both **file search patterns** and **output organization**:

### Case Mode (`saving_mode="case"`)

**File Search Pattern**:
- Transforms: `transform_folder/<PREFIX>/<PREFIX>_transformation.tfm`
- References: `reference_folder/<PREFIX>/<PREFIX>_registered.nii.gz`

**Output Structure**:
- Creates: `output_path/<PREFIX>/<PREFIX>_bbox_registered.nii.gz`

### Folder Mode (`saving_mode="folder"`)

**File Search Pattern**:
- Transforms: `transform_folder/<PREFIX>_transformation.tfm`
- References: `reference_folder/<PREFIX>_registered.nii.gz`

**Output Structure**:
- Creates: `output_path/<PREFIX>_bbox_registered.nii.gz`

## Returns

`None` — The function saves all transformed annotations to disk.

## Output Files

For each input annotation file, the function generates:

| Saving Mode | Output Pattern                                    | Description                     |
|-------------|---------------------------------------------------|---------------------------------|
| `"case"`    | `output_path/<PREFIX>/<PREFIX>_bbox_registered.nii.gz` | One folder per case       |
| `"folder"`  | `output_path/<PREFIX>_bbox_registered.nii.gz`     | All annotations in single folder      |

## Annotation Suffix Handling

The function automatically strips common annotation suffixes when matching with transforms/references:

**Supported suffixes**: `_bbox`, `_annotation`, `_lesion`, `_clot`, `.bbox`, `.annotation`, `.lesion`, `.clot`

**Example**:
```
Annotation file:  case001_bbox.nii.gz
Stripped prefix:  case001              ← Used for matching
Searches for:     case001_transformation.tfm
                  case001_registered.nii.gz
```

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | `annotation_folder` does not exist or contains no `.nii.gz` files |
| `ValueError`         | `saving_mode` is not `"case"` or `"folder"`                       |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are accepted
- **3D Annotations Required**: All inputs must be 3D NIfTI images
- **Prefix Matching**: Annotation filenames can include suffixes (automatically stripped)
- **Missing Files**: If transform or reference is not found, annotation is skipped with warning (if `debug=True`)
- **Output Directories**: Automatically created if they don't exist
- **Progress Tracking**: Uses tqdm progress bar to show processing status
- **Bounding Box Mode**: Default creates axis-aligned boxes (recommended for object detection)

## Examples

### Basic Usage - Case Mode
Process annotations after case-mode registration:

```python
from nidataset.preprocessing import register_CTA_dataset, register_annotation_dataset

# Step 1: Register all CTA images
register_CTA_dataset(
    nii_folder="data/scans/",
    mask_folder="data/brain_masks/",
    template_path="atlas/template.nii.gz",
    template_mask_path="atlas/template_mask.nii.gz",
    output_path="data/registered/",
    saving_mode="case",
    cleanup=True
)

# Step 2: Transform all lesion bounding boxes
register_annotation_dataset(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    reference_folder="data/registered/",
    output_path="data/lesion_bboxes_registered/",
    recalculate_bbox=True,
    saving_mode="case",
    debug=True
)
```

### Basic Usage - Folder Mode
Process annotations after folder-mode registration:

```python
from nidataset.preprocessing import register_annotation_dataset

register_annotation_dataset(
    annotation_folder="data/hemorrhage_bboxes/",
    transform_folder="data/registered/transforms/",
    reference_folder="data/registered/registered/",
    output_path="data/hemorrhage_bboxes_registered/",
    recalculate_bbox=True,
    saving_mode="folder",
    debug=True
)
```

### Process Multiple Annotation Types
Transform different annotation categories:

```python
from nidataset.preprocessing import register_annotation_dataset

annotation_types = ["lesion_bboxes", "vessel_bboxes", "hemorrhage_bboxes"]

for annotation_type in annotation_types:
    print(f"Processing {annotation_type}...")
    
    register_annotation_dataset(
        annotation_folder=f"data/{annotation_type}/",
        transform_folder="data/registered/",
        reference_folder="data/registered/",
        output_path=f"data/{annotation_type}_registered/",
        recalculate_bbox=True,
        saving_mode="case",
        debug=True
    )
```

### Track Registration Success Rate
Monitor processing results:

```python
from nidataset.preprocessing import register_annotation_dataset
import os

# Register annotations
register_annotation_dataset(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    reference_folder="data/registered/",
    output_path="data/lesion_bboxes_registered/",
    saving_mode="case",
    debug=False
)

# Count results
annotation_files = [f for f in os.listdir("data/lesion_bboxes/") 
                   if f.endswith(".nii.gz")]
registered_count = len([f for f in os.listdir("data/lesion_bboxes_registered/")
                       if os.path.isdir(os.path.join("data/lesion_bboxes_registered/", f))])

print(f"Success rate: {registered_count}/{len(annotation_files)} "
      f"({(registered_count/len(annotation_files))*100:.1f}%)")
```

### Create Detection Dataset with Metadata
Complete workflow with bounding box extraction:

```python
from nidataset.preprocessing import register_CTA_dataset, register_annotation_dataset
import nibabel as nib
import numpy as np
import pandas as pd
import os

# Register scans
register_CTA_dataset(
    nii_folder="data/scans/",
    mask_folder="data/brain_masks/",
    template_path="atlas/template.nii.gz",
    template_mask_path="atlas/template_mask.nii.gz",
    output_path="data/registered/",
    saving_mode="case",
    cleanup=True
)

# Register annotations
register_annotation_dataset(
    annotation_folder="data/bboxes/",
    transform_folder="data/registered/",
    reference_folder="data/registered/",
    output_path="data/bboxes_registered/",
    recalculate_bbox=True,
    saving_mode="case"
)

# Extract bbox coordinates
cases = os.listdir("data/registered/")
bbox_catalog = []

for case in cases:
    bbox_path = f"data/bboxes_registered/{case}/{case}_bbox_registered.nii.gz"
    
    if os.path.exists(bbox_path):
        bbox_data = nib.load(bbox_path).get_fdata()
        coords = np.argwhere(bbox_data > 0)
        
        if coords.size > 0:
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            
            bbox_catalog.append({
                'case_id': case,
                'x_min': int(x_min), 'y_min': int(y_min), 'z_min': int(z_min),
                'x_max': int(x_max), 'y_max': int(y_max), 'z_max': int(z_max)
            })

# Save catalog
df = pd.DataFrame(bbox_catalog)
df.to_csv("data/bbox_catalog.csv", index=False)
print(f"Created catalog with {len(df)} annotated cases")
```

## Typical Workflow

```python
from nidataset.preprocessing import register_CTA_dataset, register_annotation_dataset

# Phase 1: Register primary scans
register_CTA_dataset(
    nii_folder="data/scans/",
    mask_folder="data/brain_masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_path="data/registered/",
    saving_mode="case",
    cleanup=True
)

# Phase 2: Propagate annotations
register_annotation_dataset(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    reference_folder="data/registered/",
    output_path="data/lesion_bboxes_registered/",
    recalculate_bbox=True,
    saving_mode="case"
)

# Phase 3: Use for object detection training
```
