---
title: register_mask_dataset
parent: Package Functions
nav_order: 36
---

# `register_mask_dataset`

Apply saved transformations to all masks in a dataset folder using previously computed registration transforms.

```python
register_mask_dataset(
    mask_folder: str,
    transform_folder: str,
    registered_folder: str,
    output_path: str,
    is_binary: bool = True,
    saving_mode: str = "case",
    debug: bool = False
) -> None
```

## Overview

This function processes all mask files in a dataset by applying previously computed transformations from image registration. Each mask is transformed using its corresponding transformation file and reference registered image, enabling batch processing of segmentations, ROI masks, or label maps.

**Batch transformation pipeline**:
1. **Discovery**: Locates all masks in input folder
2. **File matching**: Finds corresponding transforms and references by prefix
3. **Transformation**: Applies saved transformations with appropriate interpolation
4. **Organization**: Saves results according to `saving_mode`

This is essential for:
- Propagating segmentation masks across entire datasets
- Applying vessel, lesion, or ROI masks to registered volumes
- Creating normalized mask collections for population studies
- Maintaining spatial correspondence in multi-modal analyses
- Building standardized mask datasets for machine learning

## Parameters

| Name                | Type   | Default    | Description                                                                                          |
|---------------------|--------|------------|------------------------------------------------------------------------------------------------------|
| `mask_folder`       | `str`  | *required* | Path to folder containing input mask files ending with `_mask.nii.gz`.                              |
| `transform_folder`  | `str`  | *required* | Path to folder containing transformation files (`.tfm`). Location depends on `saving_mode`.         |
| `registered_folder`  | `str`  | *required* | Path to folder containing registered reference images. Location depends on `saving_mode`.           |
| `output_path`       | `str`  | *required* | Base directory for all outputs. Organization depends on `saving_mode`.                              |
| `is_binary`         | `bool` | `True`     | If `True`, uses nearest neighbor interpolation. If `False`, uses linear interpolation.             |
| `saving_mode`       | `str`  | `"case"`   | Organization mode: `"case"` (per-case subfolders) or `"folder"` (single directory).               |
| `debug`             | `bool` | `False`    | If `True`, prints detailed information and warnings for each mask.                                  |

## Saving Mode Behavior

The `saving_mode` parameter controls both **file search patterns** and **output organization**:

### Case Mode (`saving_mode="case"`)

**File Search Pattern**:
- Transforms: `transform_folder/<PREFIX>/<PREFIX>_transformation.tfm`
- References: `registered_folder/<PREFIX>/<PREFIX>_registered.nii.gz`

**Output Structure**:
- Creates: `output_path/<PREFIX>/<PREFIX>_registered_mask.nii.gz`

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
│   └── case001_registered_mask.nii.gz
├── case002/
│   └── case002_registered_mask.nii.gz
└── case003/
    └── case003_registered_mask.nii.gz
```

### Folder Mode (`saving_mode="folder"`)

**File Search Pattern**:
- Transforms: `transform_folder/<PREFIX>_transformation.tfm`
- References: `registered_folder/<PREFIX>_registered.nii.gz`

**Output Structure**:
- Creates: `output_path/<PREFIX>_registered_mask.nii.gz`

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
├── case001_registered_mask.nii.gz
├── case002_registered_mask.nii.gz
└── case003_registered_mask.nii.gz
```

## Returns

`None` — The function saves all transformed masks to disk.

## Output Files

For each input mask file, the function generates:

| Saving Mode | Output Pattern                                    | Description                     |
|-------------|---------------------------------------------------|---------------------------------|
| `"case"`    | `output_path/<PREFIX>/<PREFIX>_registered_mask.nii.gz` | One folder per case       |
| `"folder"`  | `output_path/<PREFIX>_registered_mask.nii.gz`     | All masks in single folder      |

## Interpolation Selection Guide

| Mask Type | is_binary Setting | Reason |
|-----------|-------------------|--------|
| Binary segmentation | `True` | Preserves discrete labels (0, 1) |
| Multi-label segmentation | `True` | Preserves label integrity (0, 1, 2, 3, ...) |
| Vessel masks | `True` | Maintains vessel vs. background distinction |
| Lesion masks | `True` | Preserves lesion boundaries |
| Tissue probability maps | `False` | Allows smooth probability values |
| Confidence maps | `False` | Maintains continuous confidence scores |
| Gradient maps | `False` | Preserves smooth intensity transitions |
| Distance maps | `False` | Maintains continuous distance values |

## Important Notes

### Mask Naming Convention
- **All mask files must end with `_mask.nii.gz`**
- This is enforced by the function's prefix extraction logic
- Examples of valid names:
  - `patient001_brain_mask.nii.gz` (prefix: `patient001_brain`)
  - `lesion_mask.nii.gz` (prefix: `lesion`)
  - `vessel_segmentation_mask.nii.gz` (prefix: `vessel_segmentation`)
- Examples of invalid names:
  - `patient001_brain.nii.gz` (missing `_mask` suffix)
  - `lesion.nii.gz` (missing `_mask` suffix)
  - `mask_brain.nii.gz` (`_mask` not at end before `.nii.gz`)

### File Matching in Workflows
When using `register_mask_dataset`, the function will:
1. Look for masks ending with `_mask.nii.gz`
2. Strip the `_mask.nii.gz` suffix to extract the prefix
3. Match the prefix to transformations and references
4. Example: `patient001_brain_mask.nii.gz` → prefix `patient001_brain` → matches `patient001_brain_transformation.tfm` and `patient001_brain_registered.nii.gz`

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | `mask_folder` does not exist or contains no `.nii.gz` files       |
| `ValueError`         | `saving_mode` is not `"case"` or `"folder"`                       |

## Usage Notes

- **Input Format**: Only files ending with `_mask.nii.gz` are processed
- **3D Masks Required**: All inputs must be 3D NIfTI images
- **Prefix Matching**: The prefix (extracted by removing `_mask.nii.gz`) must match corresponding transform/reference prefixes
- **Missing Files**: If transform or reference is not found, the mask is skipped with a warning (if `debug=True`)
- **Output Directories**: Automatically created if they don't exist
- **Progress Tracking**: Uses tqdm progress bar to show processing status
- **Error Handling**: Continues processing remaining masks even if individual masks fail

## Examples

### Basic Usage - Case Mode
Process masks after case-mode registration:

```python
from nidataset.preprocessing import register_CTA_dataset, register_mask_dataset

# Step 1: Register all CTA images in case mode
register_CTA_dataset(
    nii_folder="data/scans/",
    mask_folder="data/brain_masks/",
    template_path="atlas/template.nii.gz",
    template_mask_path="atlas/template_mask.nii.gz",
    output_path="data/registered/",
    saving_mode="case",
    cleanup=True
)

# Step 2: Apply transformations to vessel masks
# Masks must be named like: case001_vessel_mask.nii.gz
register_mask_dataset(
    mask_folder="data/vessel_masks/",
    transform_folder="data/registered/",  # searches in <PREFIX>/ subdirs
    registered_folder="data/registered/",  # searches in <PREFIX>/ subdirs
    output_path="data/vessel_registered/",
    is_binary=True,
    saving_mode="case",
    debug=True
)
```

### Basic Usage - Folder Mode
Process masks after folder-mode registration:

```python
from nidataset.preprocessing import register_CTA_dataset, register_mask_dataset

# Step 1: Register all CTA images in folder mode
register_CTA_dataset(
    nii_folder="data/scans/",
    mask_folder="data/brain_masks/",
    template_path="atlas/template.nii.gz",
    template_mask_path="atlas/template_mask.nii.gz",
    output_path="data/registered/",
    saving_mode="folder",
    cleanup=True
)
# Creates: registered/registered/case_001_registered.nii.gz
#          registered/transforms/case_001_transformation.tfm

# Step 2: Apply transformations to lesion masks
# Masks must be named like: case_001_lesion_mask.nii.gz
register_mask_dataset(
    mask_folder="data/lesion_masks/",
    transform_folder="data/registered/transforms/",  # direct prefix search
    registered_folder="data/registered/registered/",  # direct prefix search
    output_path="data/lesion_registered/",
    is_binary=True,
    saving_mode="folder",
    debug=True
)
```

### Register Multiple Mask Types
Process different mask types sequentially:

```python
from nidataset.preprocessing import register_mask_dataset

# After registering main scans with saving_mode="case"
# Each mask type must follow naming: <prefix>_<type>_mask.nii.gz
mask_types = [
    ("lesion_masks", "lesion_registered"),
    ("vessel_masks", "vessel_registered"),
    ("hemorrhage_masks", "hemorrhage_registered"),
    ("edema_masks", "edema_registered")
]

for mask_folder_name, output_folder_name in mask_types:
    print(f"\nProcessing {mask_folder_name}...")
    
    register_mask_dataset(
        mask_folder=f"data/{mask_folder_name}/",
        transform_folder="data/registered/",
        registered_folder="data/registered/",
        output_path=f"data/{output_folder_name}/",
        is_binary=True,
        saving_mode="case",
        debug=True
    )
    
    print(f"Completed {mask_folder_name}")
```

### Register Probability Maps
Process continuous-valued masks:

```python
from nidataset.preprocessing import register_mask_dataset

# Register tissue probability maps
# Files must be named like: case001_gray_matter_prob_mask.nii.gz
probability_maps = [
    "gray_matter_prob",
    "white_matter_prob",
    "csf_prob"
]

for prob_map in probability_maps:
    print(f"\nRegistering {prob_map} maps...")
    
    register_mask_dataset(
        mask_folder=f"data/{prob_map}/",
        transform_folder="data/registered/transforms/",
        registered_folder="data/registered/registered/",
        output_path=f"data/{prob_map}_registered/",
        is_binary=False,  # Use linear interpolation
        saving_mode="folder",
        debug=True
    )
```

### Error Handling and Validation
Robust processing with validation:

```python
from nidataset.preprocessing import register_mask_dataset
import os

def validate_and_register_masks(mask_folder, transform_folder, 
                                registered_folder, output_path, 
                                saving_mode="case"):
    """Register masks with pre-validation."""
    
    # Validate input folder exists
    if not os.path.isdir(mask_folder):
        print(f"Error: Mask folder does not exist: {mask_folder}")
        return False
    
    # Count mask files (must end with _mask.nii.gz)
    mask_files = [f for f in os.listdir(mask_folder) 
                  if f.endswith("_mask.nii.gz")]
    if not mask_files:
        print(f"Error: No _mask.nii.gz files found in {mask_folder}")
        return False
    
    print(f"Found {len(mask_files)} mask files to process")
    
    # Validate transform and reference folders
    if not os.path.isdir(transform_folder):
        print(f"Error: Transform folder does not exist: {transform_folder}")
        return False
    
    if not os.path.isdir(registered_folder):
        print(f"Error: Reference folder does not exist: {registered_folder}")
        return False
    
    # Process masks
    try:
        register_mask_dataset(
            mask_folder=mask_folder,
            transform_folder=transform_folder,
            registered_folder=registered_folder,
            output_path=output_path,
            is_binary=True,
            saving_mode=saving_mode,
            debug=True
        )
        print(f"Successfully registered masks to {output_path}")
        return True
    
    except Exception as e:
        print(f"Registration failed: {str(e)}")
        return False

# Use validation function
validate_and_register_masks(
    mask_folder="data/lesion_masks/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/lesion_registered/",
    saving_mode="case"
)
```

### Tracking Registration Success Rate
Monitor processing results:

```python
from nidataset.preprocessing import register_mask_dataset
import os

def count_registered_masks(mask_folder, output_path, saving_mode="case"):
    """Count successfully registered masks."""
    
    mask_files = [f for f in os.listdir(mask_folder) 
                  if f.endswith("_mask.nii.gz")]
    total_masks = len(mask_files)
    
    registered_count = 0
    
    for mask_file in mask_files:
        # Extract prefix by removing _mask.nii.gz
        prefix = mask_file.replace("_mask.nii.gz", "")
        
        if saving_mode == "case":
            output_file = os.path.join(output_path, prefix, 
                                      f"{prefix}_registered_mask.nii.gz")
        else:
            output_file = os.path.join(output_path, 
                                      f"{prefix}_registered_mask.nii.gz")
        
        if os.path.exists(output_file):
            registered_count += 1
    
    return registered_count, total_masks

# Register masks
register_mask_dataset(
    mask_folder="data/vessel_masks/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/vessel_registered/",
    saving_mode="case",
    debug=False
)

# Check results
registered, total = count_registered_masks(
    mask_folder="data/vessel_masks/",
    output_path="data/vessel_registered/",
    saving_mode="case"
)

print(f"\nRegistration Summary:")
print(f"  Total masks: {total}")
print(f"  Successfully registered: {registered}")
print(f"  Failed/skipped: {total - registered}")
print(f"  Success rate: {(registered/total)*100:.1f}%")
```

### Complete Multi-Modal Pipeline
Full workflow with multiple mask types:

```python
from nidataset.preprocessing import register_CTA_dataset, register_mask_dataset
import os

def complete_registration_pipeline(base_dir, template_path, template_mask_path):
    """
    Complete registration pipeline for scans and multiple mask types.
    
    Directory structure expected:
    base_dir/
    ├── scans/
    ├── brain_masks/
    ├── lesion_masks/        # Files: case001_lesion_mask.nii.gz, etc.
    ├── vessel_masks/        # Files: case001_vessel_mask.nii.gz, etc.
    └── roi_masks/           # Files: case001_roi_mask.nii.gz, etc.
    """
    
    print("=" * 60)
    print("PHASE 1: Registering CTA scans")
    print("=" * 60)
    
    # Register main scans
    register_CTA_dataset(
        nii_folder=f"{base_dir}/scans/",
        mask_folder=f"{base_dir}/brain_masks/",
        template_path=template_path,
        template_mask_path=template_mask_path,
        output_path=f"{base_dir}/registered/",
        saving_mode="case",
        cleanup=True,
        debug=True
    )
    
    print("\n" + "=" * 60)
    print("PHASE 2: Registering mask datasets")
    print("=" * 60)
    
    # Define mask types to register
    mask_datasets = [
        ("lesion_masks", True, "Lesion masks"),
        ("vessel_masks", True, "Vessel masks"),
        ("roi_masks", True, "ROI masks")
    ]
    
    results = {}
    
    for folder_name, is_binary, description in mask_datasets:
        mask_folder = f"{base_dir}/{folder_name}/"
        
        # Skip if folder doesn't exist
        if not os.path.isdir(mask_folder):
            print(f"\nSkipping {description}: folder not found")
            results[folder_name] = "skipped"
            continue
        
        print(f"\nProcessing {description}...")
        
        try:
            register_mask_dataset(
                mask_folder=mask_folder,
                transform_folder=f"{base_dir}/registered/",
                registered_folder=f"{base_dir}/registered/",
                output_path=f"{base_dir}/{folder_name}_registered/",
                is_binary=is_binary,
                saving_mode="case",
                debug=True
            )
            results[folder_name] = "success"
            print(f"{description} registered successfully")
        
        except Exception as e:
            results[folder_name] = f"failed: {str(e)}"
            print(f"{description} registration failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    for dataset, status in results.items():
        print(f"{dataset}: {status}")

# Execute pipeline
complete_registration_pipeline(
    base_dir="data",
    template_path="atlas/template.nii.gz",
    template_mask_path="atlas/template_mask.nii.gz"
)
```

### Parallel Processing for Large Datasets
Process masks in parallel (requires joblib):

```python
from nidataset.preprocessing import register_mask
from joblib import Parallel, delayed
import os
from tqdm import tqdm

def process_single_mask(mask_file, mask_folder, transform_folder, 
                        registered_folder, output_path, saving_mode):
    """Process a single mask file."""
    
    # Extract prefix by removing _mask.nii.gz
    prefix = mask_file.replace("_mask.nii.gz", "")
    mask_path = os.path.join(mask_folder, mask_file)
    
    # Find corresponding files based on saving mode
    if saving_mode == "case":
        transform_file = os.path.join(transform_folder, prefix, 
                                     f"{prefix}_transformation.tfm")
        reference_file = os.path.join(registered_folder, prefix, 
                                     f"{prefix}_registered.nii.gz")
        output_file = os.path.join(output_path, prefix, 
                                  f"{prefix}_registered_mask.nii.gz")
        os.makedirs(os.path.join(output_path, prefix), exist_ok=True)
    else:
        transform_file = os.path.join(transform_folder, 
                                     f"{prefix}_transformation.tfm")
        reference_file = os.path.join(registered_folder, 
                                     f"{prefix}_registered.nii.gz")
        output_file = os.path.join(output_path, 
                                  f"{prefix}_registered_mask.nii.gz")
    
    # Check if required files exist
    if not os.path.exists(transform_file) or not os.path.exists(reference_file):
        return prefix, "skipped"
    
    try:
        register_mask(
            mask_path=mask_path,
            transform_path=transform_file,
            registered_path=reference_file,
            output_path=output_file,
            is_binary=True,
            debug=False
        )
        return prefix, "success"
    except Exception as e:
        return prefix, f"failed: {str(e)}"

def parallel_register_masks(mask_folder, transform_folder, registered_folder,
                            output_path, saving_mode="case", n_jobs=-1):
    """Register masks in parallel."""
    
    # Get mask files (must end with _mask.nii.gz)
    mask_files = [f for f in os.listdir(mask_folder) 
                  if f.endswith("_mask.nii.gz")]
    
    if not mask_files:
        print("No mask files found")
        return
    
    print(f"Processing {len(mask_files)} masks with {n_jobs} parallel jobs...")
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_mask)(
            mask_file, mask_folder, transform_folder, 
            registered_folder, output_path, saving_mode
        )
        for mask_file in tqdm(mask_files, desc="Registering masks")
    )
    
    # Summarize results
    success_count = sum(1 for _, status in results if status == "success")
    
    print(f"\nProcessing complete:")
    print(f"  Total: {len(results)}")
    print(f"  Success: {success_count}")
    print(f"  Failed/Skipped: {len(results) - success_count}")
    
    return results

# Use parallel processing
results = parallel_register_masks(
    mask_folder="data/vessel_masks/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/vessel_registered/",
    saving_mode="case",
    n_jobs=8  # Use 8 CPU cores
)
```

### Quality Control After Batch Registration
Verify registered masks:

```python
from nidataset.preprocessing import register_mask_dataset
import nibabel as nib
import numpy as np
import os

# Register masks
register_mask_dataset(
    mask_folder="data/lesion_masks/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/lesion_registered/",
    saving_mode="case",
    debug=True
)

# Quality control
def check_registered_masks(output_path, registered_folder, saving_mode="case"):
    """Verify registered mask quality."""
    
    cases = [d for d in os.listdir(output_path) 
             if os.path.isdir(os.path.join(output_path, d))]
    
    qc_results = []
    
    for case in cases:
        if saving_mode == "case":
            mask_path = os.path.join(output_path, case, 
                                    f"{case}_registered_mask.nii.gz")
            ref_path = os.path.join(registered_folder, case, 
                                   f"{case}_registered.nii.gz")
        else:
            mask_path = os.path.join(output_path, 
                                    f"{case}_registered_mask.nii.gz")
            ref_path = os.path.join(registered_folder, 
                                   f"{case}_registered.nii.gz")
        
        if not os.path.exists(mask_path) or not os.path.exists(ref_path):
            continue
        
        # Load data
        mask = nib.load(mask_path).get_fdata()
        reference = nib.load(ref_path).get_fdata()
        
        # Check shape match
        shapes_match = mask.shape == reference.shape
        
        # Check mask coverage
        mask_coverage = (np.sum(mask > 0) / mask.size) * 100
        
        # Check if mask is empty
        is_empty = np.sum(mask) == 0
        
        qc_results.append({
            'case': case,
            'shapes_match': shapes_match,
            'coverage_percent': mask_coverage,
            'is_empty': is_empty,
            'passed': shapes_match and not is_empty and mask_coverage > 0.1
        })
    
    return qc_results

# Run QC
qc_results = check_registered_masks(
    output_path="data/lesion_registered/",
    registered_folder="data/registered/",
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
    print(f"  Shapes match: {result['shapes_match']}")
    print(f"  Coverage: {result['coverage_percent']:.2f}%")
    print(f"  Empty: {result['is_empty']}")

print(f"\n{'=' * 60}")
print(f"Overall: {passed_count}/{len(qc_results)} cases passed QC")
```

### Converting Between Saving Modes
Reorganize registered masks:

```python
import os
import shutil
from pathlib import Path

def convert_case_to_folder(input_path, output_path):
    """Convert case-mode organization to folder-mode."""
    
    os.makedirs(output_path, exist_ok=True)
    
    cases = [d for d in os.listdir(input_path) 
             if os.path.isdir(os.path.join(input_path, d))]
    
    for case in cases:
        case_dir = os.path.join(input_path, case)
        
        # Find registered mask
        mask_file = f"{case}_registered_mask.nii.gz"
        src_path = os.path.join(case_dir, mask_file)
        
        if os.path.exists(src_path):
            dst_path = os.path.join(output_path, mask_file)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {case}")
    
    print(f"\nConverted {len(cases)} cases from case-mode to folder-mode")

# Convert organization
convert_case_to_folder(
    input_path="data/lesion_registered/",
    output_path="data/lesion_registered_flat/"
)
```

### Integration with Analysis Pipeline
Use registered masks for statistical analysis:

```python
from nidataset.preprocessing import register_CTA_dataset, register_mask_dataset
import nibabel as nib
import numpy as np
import pandas as pd
import os

# Step 1: Register scans and masks
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

print("\nRegistering lesion masks...")
# Lesion masks must be named: case001_lesion_mask.nii.gz, etc.
register_mask_dataset(
    mask_folder="data/lesion_masks/",
    transform_folder="data/registered/",
    registered_folder="data/registered/",
    output_path="data/lesion_registered/",
    saving_mode="case"
)

# Step 2: Extract statistics from registered data
def extract_lesion_statistics(registered_folder, lesion_folder, cases):
    """Extract statistics from registered scans and lesion masks."""
    
    statistics = []
    
    for case in cases:
        # Load registered scan and lesion mask
        scan_path = os.path.join(registered_folder, case, 
                                f"{case}_registered.nii.gz")
        lesion_path = os.path.join(lesion_folder, case, 
                                  f"{case}_registered_mask.nii.gz")
        
        if not os.path.exists(scan_path) or not os.path.exists(lesion_path):
            continue
        
        scan = nib.load(scan_path).get_fdata()
        lesion = nib.load(lesion_path).get_fdata()
        
        # Calculate statistics
        lesion_voxels = scan[lesion > 0]
        healthy_voxels = scan[lesion == 0]
        
        stats = {
            'case': case,
            'lesion_volume_voxels': np.sum(lesion > 0),
            'lesion_mean_intensity': np.mean(lesion_voxels) if len(lesion_voxels) > 0 else 0,
            'lesion_std_intensity': np.std(lesion_voxels) if len(lesion_voxels) > 0 else 0,
            'healthy_mean_intensity': np.mean(healthy_voxels) if len(healthy_voxels) > 0 else 0,
            'intensity_contrast': (np.mean(lesion_voxels) - np.mean(healthy_voxels)) 
                                 if len(lesion_voxels) > 0 and len(healthy_voxels) > 0 else 0
        }
        
        statistics.append(stats)
    
    return pd.DataFrame(statistics)

# Extract statistics
cases = ["case001", "case002", "case003"]
df = extract_lesion_statistics(
    registered_folder="data/registered/",
    lesion_folder="data/lesion_registered/",
    cases=cases
)

# Display results
print("\n" + "=" * 60)
print("LESION STATISTICS")
print("=" * 60)
print(df.to_string(index=False))

# Save to CSV
df.to_csv("lesion_statistics.csv", index=False)
print("\nStatistics saved to lesion_statistics.