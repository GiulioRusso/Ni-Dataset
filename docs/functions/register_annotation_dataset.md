---
title: register_annotation_dataset
parent: Package Functions
nav_order: 18
---

# `register_annotation_dataset`

Apply saved transformations to all annotations in a dataset folder using previously computed registration transforms.

```python
register_annotation_dataset(
    annotation_folder: str,
    transform_folder: str,
    registered_folder: str,
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
2. **Prefix matching**: Finds corresponding transforms and references by stripping `_bbox.nii.gz`
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
| `annotation_folder` | `str`  | *required* | Path to folder containing input annotation files ending with `_bbox.nii.gz`.                        |
| `transform_folder`  | `str`  | *required* | Path to folder containing transformation files (`.tfm`). Location depends on `saving_mode`.         |
| `registered_folder`  | `str`  | *required* | Path to folder containing registered reference images. Location depends on `saving_mode`.           |
| `output_path`       | `str`  | *required* | Base directory for all outputs. Organization depends on `saving_mode`.                              |
| `recalculate_bbox`  | `bool` | `True`     | If `True`, creates axis-aligned boxes. If `False`, preserves deformed shapes.                      |
| `saving_mode`       | `str`  | `"case"`   | Organization mode: `"case"` (per-case subfolders) or `"folder"` (single directory).               |
| `debug`             | `bool` | `False`    | If `True`, prints detailed information and warnings for each annotation.                            |

## Saving Mode Behavior

The `saving_mode` parameter controls both **file search patterns** and **output organization**:

### Case Mode (`saving_mode="case"`)

**File Search Pattern**:
- Transforms: `transform_folder/<PREFIX>/<PREFIX>_transformation.tfm`
- References: `registered_folder/<PREFIX>/<PREFIX>_registered.nii.gz`

**Output Structure**:
- Creates: `output_path/<PREFIX>/<PREFIX>_registered_bbox.nii.gz`

**Use Case**: When registration was done with `register_CTA_dataset(..., saving_mode="case")`

```
transform_folder/
├── case001/
│   └── case001_transformation.tfm
├── case002/
│   └── case002_transformation.tfm
└── case003/
    └── case003_transformation.tfm

registered_folder/
├── case001/
│   └── case001_registered.nii.gz
├── case002/
│   └── case002_registered.nii.gz
└── case003/
    └── case003_registered.nii.gz

output_path/
├── case001/
│   └── case001_registered_bbox.nii.gz
├── case002/
│   └── case002_registered_bbox.nii.gz
└── case003/
    └── case003_registered_bbox.nii.gz
```

### Folder Mode (`saving_mode="folder"`)

**File Search Pattern**:
- Transforms: `transform_folder/<PREFIX>_transformation.tfm`
- References: `registered_folder/<PREFIX>_registered.nii.gz`

**Output Structure**:
- Creates: `output_path/<PREFIX>_registered_bbox.nii.gz`

**Use Case**: When registration was done with `register_CTA_dataset(..., saving_mode="folder")`

```
transform_folder/
├── case001_transformation.tfm
├── case002_transformation.tfm
└── case003_transformation.tfm

registered_folder/
├── case001_registered.nii.gz
├── case002_registered.nii.gz
└── case003_registered.nii.gz

output_path/
├── case001_registered_bbox.nii.gz
├── case002_registered_bbox.nii.gz
└── case003_registered_bbox.nii.gz
```

## Returns

`None` — The function saves all transformed annotations to disk.

## Output Files

For each input annotation file, the function generates:

| Saving Mode | Output Pattern                                    | Description                     |
|-------------|---------------------------------------------------|---------------------------------|
| `"case"`    | `output_path/<PREFIX>/<PREFIX>_registered_bbox.nii.gz` | One folder per case       |
| `"folder"`  | `output_path/<PREFIX>_registered_bbox.nii.gz`     | All annotations in single folder      |

## Important Notes

### Annotation Naming Convention
- **All annotation files must end with `_bbox.nii.gz`**
- This is enforced by the function's prefix extraction logic
- Examples of valid names:
  - `patient001_lesion_bbox.nii.gz` (prefix: `patient001_lesion`)
  - `case001_bbox.nii.gz` (prefix: `case001`)
  - `hemorrhage_region_bbox.nii.gz` (prefix: `hemorrhage_region`)
- Examples of invalid names:
  - `patient001_lesion.nii.gz` (missing `_bbox` suffix)
  - `lesion.nii.gz` (missing `_bbox` suffix)
  - `bbox_lesion.nii.gz` (`_bbox` not at end before `.nii.gz`)

### File Matching in Workflows
When using `register_annotation_dataset`, the function will:
1. Look for annotations ending with `_bbox.nii.gz`
2. Strip the `_bbox.nii.gz` suffix to extract the prefix
3. Match the prefix to transformations and references
4. Example: `patient001_lesion_bbox.nii.gz` → prefix `patient001_lesion` → matches `patient001_lesion_transformation.tfm` and `patient001_lesion_registered.nii.gz`

### Bounding Box Recalculation
- `recalculate_bbox=True` (default): Creates new axis-aligned boxes around transformed regions
- `recalculate_bbox=False`: Preserves exact deformed shapes after transformation
- Recommended to use `True` for object detection workflows requiring rectangular boxes

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | `annotation_folder` does not exist or contains no `.nii.gz` files |
| `ValueError`         | `saving_mode` is not `"case"` or `"folder"`                       |

## Usage Notes

- **Input Format**: Only files ending with `_bbox.nii.gz` are processed
- **3D Annotations Required**: All inputs must be 3D NIfTI images
- **Prefix Matching**: The prefix (extracted by removing `_bbox.nii.gz`) must match corresponding transform/reference prefixes
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
# Annotations must be named like: case001_lesion_bbox.nii.gz
register_annotation_dataset(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
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

# Annotations must be named like: case001_hemorrhage_bbox.nii.gz
register_annotation_dataset(
    annotation_folder="data/hemorrhage_bboxes/",
    transform_folder="data/registered/transforms/",
    registered_folder="data/registered/registered/",
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

# Each annotation type must follow naming: <prefix>_<type>_bbox.nii.gz
annotation_types = [
    ("lesion_bboxes", "lesion_bboxes_registered"),
    ("vessel_bboxes", "vessel_bboxes_registered"),
    ("hemorrhage_bboxes", "hemorrhage_bboxes_registered")
]

for annotation_folder, output_folder in annotation_types:
    print(f"Processing {annotation_folder}...")
    
    register_annotation_dataset(
        annotation_folder=f"data/{annotation_folder}/",
        transform_folder="data/registered/",
        registered_folder="data/registered/",
        output_path=f"data/{output_folder}/",
        recalculate_bbox=True,
        saving_mode="case",
        debug=True
    )
```

### Compare Recalculation vs Preservation
Process same annotations with different settings:

```python
from nidataset.preprocessing import register_annotation_dataset

# With bounding box recalculation (axis-aligned)
register_annotation_dataset(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/lesion_bboxes_recalculated/",
    recalculate_bbox=True,
    saving_mode="case"
)

# Without recalculation (preserves deformation)
register_annotation_dataset(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/lesion_bboxes_deformed/",
    recalculate_bbox=False,
    saving_mode="case"
)
```

### Track Registration Success Rate
Monitor processing results:

```python
from nidataset.preprocessing import register_annotation_dataset
import os

def count_registered_annotations(annotation_folder, output_path, saving_mode="case"):
    """Count successfully registered annotations."""
    
    annotation_files = [f for f in os.listdir(annotation_folder) 
                       if f.endswith("_bbox.nii.gz")]
    total_annotations = len(annotation_files)
    
    registered_count = 0
    
    for annotation_file in annotation_files:
        # Extract prefix by removing _bbox.nii.gz
        prefix = annotation_file.replace("_bbox.nii.gz", "")
        
        if saving_mode == "case":
            output_file = os.path.join(output_path, prefix, 
                                      f"{prefix}_registered_bbox.nii.gz")
        else:
            output_file = os.path.join(output_path, 
                                      f"{prefix}_registered_bbox.nii.gz")
        
        if os.path.exists(output_file):
            registered_count += 1
    
    return registered_count, total_annotations

# Register annotations
register_annotation_dataset(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/lesion_bboxes_registered/",
    saving_mode="case",
    debug=False
)

# Check results
registered, total = count_registered_annotations(
    annotation_folder="data/lesion_bboxes/",
    output_path="data/lesion_bboxes_registered/",
    saving_mode="case"
)

print(f"\nRegistration Summary:")
print(f"  Total annotations: {total}")
print(f"  Successfully registered: {registered}")
print(f"  Failed/skipped: {total - registered}")
print(f"  Success rate: {(registered/total)*100:.1f}%")
```

### Create Detection Dataset with Metadata
Complete workflow with bounding box extraction:

```python
from nidataset.preprocessing import register_CTA_dataset, register_annotation_dataset
import nibabel as nib
import numpy as np
import pandas as pd
import os

# Step 1: Register scans
print("Registering scans...")
register_CTA_dataset(
    nii_folder="data/scans/",
    mask_folder="data/brain_masks/",
    template_path="atlas/template.nii.gz",
    template_mask_path="atlas/template_mask.nii.gz",
    output_path="data/registered/",
    saving_mode="case",
    cleanup=True
)

# Step 2: Register annotations
# Annotations must be named: case001_lesion_bbox.nii.gz, etc.
print("\nRegistering annotations...")
register_annotation_dataset(
    annotation_folder="data/bboxes/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/bboxes_registered/",
    recalculate_bbox=True,
    saving_mode="case"
)

# Step 3: Extract bbox coordinates
print("\nExtracting bounding box coordinates...")
cases = [d for d in os.listdir("data/registered/") 
         if os.path.isdir(os.path.join("data/registered/", d))]
bbox_catalog = []

for case in cases:
    bbox_path = f"data/bboxes_registered/{case}/{case}_registered_bbox.nii.gz"
    
    if os.path.exists(bbox_path):
        bbox_data = nib.load(bbox_path).get_fdata()
        coords = np.argwhere(bbox_data > 0)
        
        if coords.size > 0:
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            
            bbox_catalog.append({
                'case_id': case,
                'x_min': int(x_min), 'y_min': int(y_min), 'z_min': int(z_min),
                'x_max': int(x_max), 'y_max': int(y_max), 'z_max': int(z_max),
                'width': int(x_max - x_min + 1),
                'height': int(y_max - y_min + 1),
                'depth': int(z_max - z_min + 1)
            })

# Save catalog
df = pd.DataFrame(bbox_catalog)
df.to_csv("data/bbox_catalog.csv", index=False)

print(f"\nCreated catalog with {len(df)} annotated cases")
print(f"\nBounding box statistics:")
print(f"  Mean width: {df['width'].mean():.1f} voxels")
print(f"  Mean height: {df['height'].mean():.1f} voxels")
print(f"  Mean depth: {df['depth'].mean():.1f} voxels")
```

### Validate Annotation Names
Check that all annotations follow the naming convention:

```python
import os

def validate_annotation_names(annotation_folder):
    """Check that all annotation files end with _bbox.nii.gz."""
    
    all_files = [f for f in os.listdir(annotation_folder) 
                 if f.endswith(".nii.gz")]
    valid_files = [f for f in all_files if f.endswith("_bbox.nii.gz")]
    invalid_files = [f for f in all_files if not f.endswith("_bbox.nii.gz")]
    
    print(f"Validation Results for {annotation_folder}:")
    print(f"  Total .nii.gz files: {len(all_files)}")
    print(f"  Valid (_bbox.nii.gz): {len(valid_files)}")
    print(f"  Invalid: {len(invalid_files)}")
    
    if invalid_files:
        print("\nInvalid files (missing _bbox suffix):")
        for f in invalid_files:
            print(f"  - {f}")
        return False
    else:
        print("\nAll files are valid!")
        return True

# Validate before processing
if validate_annotation_names("data/lesion_bboxes/"):
    register_annotation_dataset(
        annotation_folder="data/lesion_bboxes/",
        transform_folder="data/registered/",
        registered_folder="data/registered/",
        output_path="data/lesion_bboxes_registered/",
        saving_mode="case"
    )
```

### Parallel Processing for Large Datasets
Process annotations in parallel (requires joblib):

```python
from nidataset.preprocessing import register_annotation
from joblib import Parallel, delayed
import os
from tqdm import tqdm

def process_single_annotation(annotation_file, annotation_folder, 
                              transform_folder, registered_folder, 
                              output_path, saving_mode, recalculate_bbox):
    """Process a single annotation file."""
    
    # Extract prefix by removing _bbox.nii.gz
    prefix = annotation_file.replace("_bbox.nii.gz", "")
    annotation_path = os.path.join(annotation_folder, annotation_file)
    
    # Find corresponding files based on saving mode
    if saving_mode == "case":
        transform_file = os.path.join(transform_folder, prefix, 
                                     f"{prefix}_transformation.tfm")
        reference_file = os.path.join(registered_folder, prefix, 
                                     f"{prefix}_registered.nii.gz")
        output_file = os.path.join(output_path, prefix, 
                                  f"{prefix}_registered_bbox.nii.gz")
        os.makedirs(os.path.join(output_path, prefix), exist_ok=True)
    else:
        transform_file = os.path.join(transform_folder, 
                                     f"{prefix}_transformation.tfm")
        reference_file = os.path.join(registered_folder, 
                                     f"{prefix}_registered.nii.gz")
        output_file = os.path.join(output_path, 
                                  f"{prefix}_registered_bbox.nii.gz")
    
    # Check if required files exist
    if not os.path.exists(transform_file) or not os.path.exists(reference_file):
        return prefix, "skipped"
    
    try:
        register_annotation(
            annotation_path=annotation_path,
            transform_path=transform_file,
            registered_path=reference_file,
            output_path=output_file,
            recalculate_bbox=recalculate_bbox,
            debug=False
        )
        return prefix, "success"
    except Exception as e:
        return prefix, f"failed: {str(e)}"

def parallel_register_annotations(annotation_folder, transform_folder, 
                                  registered_folder, output_path,
                                  saving_mode="case", recalculate_bbox=True, 
                                  n_jobs=-1):
    """Register annotations in parallel."""
    
    # Get annotation files (must end with _bbox.nii.gz)
    annotation_files = [f for f in os.listdir(annotation_folder) 
                       if f.endswith("_bbox.nii.gz")]
    
    if not annotation_files:
        print("No annotation files found")
        return
    
    print(f"Processing {len(annotation_files)} annotations with {n_jobs} parallel jobs...")
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_annotation)(
            annotation_file, annotation_folder, transform_folder, 
            registered_folder, output_path, saving_mode, recalculate_bbox
        )
        for annotation_file in tqdm(annotation_files, desc="Registering annotations")
    )
    
    # Summarize results
    success_count = sum(1 for _, status in results if status == "success")
    
    print(f"\nProcessing complete:")
    print(f"  Total: {len(results)}")
    print(f"  Success: {success_count}")
    print(f"  Failed/Skipped: {len(results) - success_count}")
    
    return results

# Use parallel processing
results = parallel_register_annotations(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/lesion_bboxes_registered/",
    saving_mode="case",
    recalculate_bbox=True,
    n_jobs=8  # Use 8 CPU cores
)
```

### Quality Control After Batch Registration
Verify registered annotations:

```python
from nidataset.preprocessing import register_annotation_dataset
import nibabel as nib
import numpy as np
import os

# Register annotations
register_annotation_dataset(
    annotation_folder="data/lesion_bboxes/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/lesion_bboxes_registered/",
    saving_mode="case",
    debug=True
)

# Quality control
def check_registered_annotations(output_path, saving_mode="case"):
    """Verify registered annotation quality."""
    
    if saving_mode == "case":
        cases = [d for d in os.listdir(output_path) 
                 if os.path.isdir(os.path.join(output_path, d))]
    else:
        cases = [f.replace("_registered_bbox.nii.gz", "") 
                 for f in os.listdir(output_path) 
                 if f.endswith("_registered_bbox.nii.gz")]
    
    qc_results = []
    
    for case in cases:
        if saving_mode == "case":
            bbox_path = os.path.join(output_path, case, 
                                    f"{case}_registered_bbox.nii.gz")
        else:
            bbox_path = os.path.join(output_path, 
                                    f"{case}_registered_bbox.nii.gz")
        
        if not os.path.exists(bbox_path):
            continue
        
        # Load annotation
        bbox_data = nib.load(bbox_path).get_fdata()
        
        # Check if empty
        is_empty = np.sum(bbox_data > 0) == 0
        
        # Get bbox size if not empty
        if not is_empty:
            coords = np.argwhere(bbox_data > 0)
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            volume = (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1)
        else:
            volume = 0
        
        qc_results.append({
            'case': case,
            'is_empty': is_empty,
            'volume_voxels': volume,
            'passed': not is_empty
        })
    
    return qc_results

# Run QC
qc_results = check_registered_annotations(
    output_path="data/lesion_bboxes_registered/",
    saving_mode="case"
)

# Print results
print("\n" + "=" * 60)
print("QUALITY CONTROL RESULTS")
print("=" * 60)

passed_count = sum(1 for r in qc_results if r['passed'])

for result in qc_results:
    status = "PASS" if result['passed'] else "FAIL"
    print(f"\n{status} - {result['case']}")
    print(f"  Empty: {result['is_empty']}")
    print(f"  Volume: {result['volume_voxels']} voxels")

print(f"\n{'=' * 60}")
print(f"Overall: {passed_count}/{len(qc_results)} cases passed QC")
```

### Complete Object Detection Pipeline
Full workflow for detection model training:

```python
from nidataset.preprocessing import register_CTA_dataset, register_annotation_dataset
import nibabel as nib
import numpy as np
import json
import os

def create_detection_dataset(base_dir, template_path, template_mask_path):
    """
    Create complete object detection dataset.
    
    Expected structure:
    base_dir/
    ├── scans/
    ├── masks/
    └── annotations/  (files: case001_lesion_bbox.nii.gz, etc.)
    """
    
    print("=" * 60)
    print("PHASE 1: Register scans")
    print("=" * 60)
    
    register_CTA_dataset(
        nii_folder=f"{base_dir}/scans/",
        mask_folder=f"{base_dir}/masks/",
        template_path=template_path,
        template_mask_path=template_mask_path,
        output_path=f"{base_dir}/registered/",
        saving_mode="case",
        cleanup=True,
        debug=True
    )
    
    print("\n" + "=" * 60)
    print("PHASE 2: Register annotations")
    print("=" * 60)
    
    register_annotation_dataset(
        annotation_folder=f"{base_dir}/annotations/",
        transform_folder=f"{base_dir}/registered/",
        registered_folder=f"{base_dir}/registered/",
        output_path=f"{base_dir}/annotations_registered/",
        recalculate_bbox=True,
        saving_mode="case",
        debug=True
    )
    
    print("\n" + "=" * 60)
    print("PHASE 3: Extract metadata")
    print("=" * 60)
    
    # Get cases
    cases = [d for d in os.listdir(f"{base_dir}/registered/") 
             if os.path.isdir(os.path.join(f"{base_dir}/registered/", d))]
    
    dataset_info = {"images": [], "annotations": []}
    
    for case in cases:
        bbox_path = f"{base_dir}/annotations_registered/{case}/{case}_registered_bbox.nii.gz"
        
        if not os.path.exists(bbox_path):
            continue
        
        # Load bbox
        bbox_data = nib.load(bbox_path).get_fdata()
        coords = np.argwhere(bbox_data > 0)
        
        if coords.size > 0:
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            
            dataset_info["images"].append({
                "id": case,
                "file_name": f"registered/{case}/{case}_registered.nii.gz"
            })
            
            dataset_info["annotations"].append({
                "image_id": case,
                "bbox": [int(x_min), int(y_min), int(z_min),
                        int(x_max), int(y_max), int(z_max)],
                "area": int((x_max - x_min) * (y_max - y_min) * (z_max - z_min))
            })
    
    # Save metadata
    with open(f"{base_dir}/detection_dataset.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset created:")
    print(f"  Total images: {len(dataset_info['images'])}")
    print(f"  Total annotations: {len(dataset_info['annotations'])}")
    print(f"  Metadata saved: {base_dir}/detection_dataset.json")

# Create dataset
create_detection_dataset(
    base_dir="detection_project",
    template_path="atlas/template.nii.gz",
    template_mask_path="atlas/template_mask.nii.gz"
)
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
    registered_folder="data/registered/",
    output_path="data/lesion_bboxes_registered/",
    recalculate_bbox=True,
    saving_mode="case"
)

# Phase 3: Use for object detection training
# - Extract bounding box coordinates
# - Create training/validation splits
# - Train detection models
```
