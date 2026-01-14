---
title: resampling_dataset
parent: Package Functions
nav_order: 24
---

# `resampling_dataset`

Batch resample all 3D volumes in a dataset folder to a uniform target size while preserving the field of view through automatic voxel spacing adjustment.

```python
resampling_dataset(
    nii_folder: str,
    output_path: str,
    desired_volume: tuple,
    saving_mode: str = "case",
    debug: bool = False
) -> None
```

## Overview

This function processes all NIfTI volumes in a dataset folder by resampling them to a consistent target size. Unlike simple resizing, resampling automatically calculates new voxel spacing to maintain the original field of view, ensuring that anatomical dimensions are preserved in physical space.

The resampling process:
1. **Calculates new spacing**: Adjusts voxel dimensions to match target size
2. **Applies B-spline interpolation**: Smooth, high-quality resampling
3. **Preserves field of view**: Maintains physical dimensions of anatomy
4. **Standardizes dimensions**: Ensures all volumes have identical array sizes

This is essential for:
- Creating uniformly sized datasets for deep learning
- Standardizing input dimensions across different scanners
- Reducing computational requirements with smaller volumes
- Enabling batch processing with fixed-size inputs
- Preparing data for convolutional neural networks

## Parameters

| Name             | Type    | Default    | Description                                                                                          |
|------------------|---------|------------|------------------------------------------------------------------------------------------------------|
| `nii_folder`     | `str`   | *required* | Path to the directory containing input volumes in `.nii.gz` format.                                 |
| `output_path`    | `str`   | *required* | Root directory where resampled volumes will be saved.                                               |
| `desired_volume` | `tuple` | *required* | Target volume size as `(X, Y, Z)`. Must contain exactly three integers.                            |
| `saving_mode`    | `str`   | `"case"`   | Organization mode: `"case"` (folder per volume) or `"folder"` (shared folder).                     |
| `debug`          | `bool`  | `False`    | If `True`, prints processing summary after completion.                                              |

## Returns

`None` – The function saves resampled volumes to disk.

## Output Organization

### Saving Modes

#### Case Mode (`saving_mode="case"`)
Creates a separate folder for each volume:
```
output_path/
├── patient_001/
│   └── patient_001_resampled.nii.gz
├── patient_002/
│   └── patient_002_resampled.nii.gz
└── patient_003/
    └── patient_003_resampled.nii.gz
```

#### Folder Mode (`saving_mode="folder"`)
All resampled volumes in a single directory:
```
output_path/
├── patient_001_resampled.nii.gz
├── patient_002_resampled.nii.gz
└── patient_003_resampled.nii.gz
```

### Filename Pattern
Each resampled volume is saved as:
```
<PREFIX>_resampled.nii.gz
```

**Example**: Input `scan_042.nii.gz` → Output `scan_042_resampled.nii.gz`

## Resampling vs Resizing

**Resampling** (this function):
- Calculates new voxel spacing automatically
- Preserves physical dimensions (field of view)
- Maintains anatomical relationships
- Uses high-quality B-spline interpolation

**Simple Resizing**:
- Just changes array dimensions
- Distorts physical space if not careful
- May not preserve voxel spacing properly

**Example**:
```
Original: 512×512×300, spacing 0.5×0.5×1.0 mm → FOV: 256×256×300 mm
Resampled to 256×256×128 → spacing 1.0×1.0×2.34 mm → FOV: 256×256×300 mm (preserved!)
```

## Target Volume Selection

Choose `desired_volume` based on your needs:

| Target Size     | Use Case                              | Memory/Speed   |
|-----------------|---------------------------------------|----------------|
| (128, 128, 128) | Fast training, initial experiments    | Low            |
| (224, 224, 128) | Balanced quality and performance      | Medium         |
| (256, 256, 256) | High quality, detailed analysis       | High           |
| (512, 512, 256) | Maximum detail, clinical applications | Very High      |

**Considerations**:
- **Smaller volumes**: Faster processing, less detail
- **Larger volumes**: More detail, higher computational cost
- **Anisotropic sizes**: Common for medical imaging (e.g., different Z resolution)
- **Network requirements**: Match input dimensions of your model

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |
| `ValueError`         | `desired_volume` does not contain exactly 3 values                 |
| `ValueError`         | Invalid `saving_mode` parameter                                    |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **3D Volumes Required**: Input must be 3D NIfTI images
- **B-spline Interpolation**: Provides smooth, high-quality results
- **Field of View**: Automatically preserved through spacing adjustment
- **Sequential Processing**: Volumes processed one at a time
- **Progress Display**: Shows progress bar during processing
- **Output Directories**: Automatically created if they don't exist

## Examples

### Basic Usage
Resample dataset to standard size:

```python
from nidataset.preprocessing import resampling_dataset

resampling_dataset(
    nii_folder="dataset/original_scans/",
    output_path="dataset/resampled/",
    desired_volume=(224, 224, 128),
    saving_mode="case"
)
# Creates: dataset/resampled/scan_001/scan_001_resampled.nii.gz, ...
```

### Folder Mode Organization
All outputs in single directory:

```python
resampling_dataset(
    nii_folder="raw_data/",
    output_path="preprocessed/uniform_size/",
    desired_volume=(256, 256, 256),
    saving_mode="folder",
    debug=True
)
# Prints: Resampling completed for all files in 'raw_data/'
```

### Small Size for Fast Training
Reduce volume size for rapid experimentation:

```python
resampling_dataset(
    nii_folder="training_data/full_res/",
    output_path="training_data/low_res/",
    desired_volume=(128, 128, 128),
    saving_mode="case",
    debug=True
)
# Smaller volumes for faster training iterations
```

### High Resolution for Analysis
Maintain detail for clinical applications:

```python
resampling_dataset(
    nii_folder="clinical_scans/",
    output_path="clinical_scans_processed/",
    desired_volume=(512, 512, 256),
    saving_mode="case",
    debug=True
)
# High resolution preserved for diagnostic accuracy
```

### Anisotropic Resampling
Different resolution in Z-axis (common in medical imaging):

```python
resampling_dataset(
    nii_folder="ct_scans/",
    output_path="ct_resampled/",
    desired_volume=(224, 224, 96),  # Lower Z resolution
    saving_mode="folder",
    debug=True
)
# Matches typical slice thickness differences
```

### Quality Control Verification
Resample and verify dimensions:

```python
import nibabel as nib
from nidataset.preprocessing import resampling_dataset

target_size = (224, 224, 128)

# Resample dataset
resampling_dataset(
    nii_folder="qa/originals/",
    output_path="qa/resampled/",
    desired_volume=target_size,
    saving_mode="folder",
    debug=True
)

# Verify results
for filename in os.listdir("qa/resampled/"):
    if filename.endswith('_resampled.nii.gz'):
        img = nib.load(f"qa/resampled/{filename}")
        
        print(f"\n{filename}:")
        print(f"  Shape: {img.shape}")
        print(f"  Target: {target_size}")
        print(f"  Match: {img.shape == target_size}")
        print(f"  Voxel spacing: {img.header.get_zooms()[:3]}")
```

### Comparing Before and After
Analyze resampling effects:

```python
import nibabel as nib
import numpy as np
from nidataset.preprocessing import resampling_dataset

# Resample
resampling_dataset(
    nii_folder="comparison/original/",
    output_path="comparison/resampled/",
    desired_volume=(256, 256, 128),
    saving_mode="folder",
    debug=True
)

# Compare a sample
original = nib.load("comparison/original/sample.nii.gz")
resampled = nib.load("comparison/resampled/sample_resampled.nii.gz")

orig_data = original.get_fdata()
resamp_data = resampled.get_fdata()

print("\nComparison:")
print(f"  Original shape: {orig_data.shape}")
print(f"  Original spacing: {original.header.get_zooms()[:3]}")
print(f"  Original FOV: {np.array(orig_data.shape) * np.array(original.header.get_zooms()[:3])}")

print(f"\n  Resampled shape: {resamp_data.shape}")
print(f"  Resampled spacing: {resampled.header.get_zooms()[:3]}")
print(f"  Resampled FOV: {np.array(resamp_data.shape) * np.array(resampled.header.get_zooms()[:3])}")

print(f"\n  FOV preserved: Field of view should be approximately equal")
```

### Multi-Resolution Processing
Create datasets at multiple resolutions:

```python
from nidataset.preprocessing import resampling_dataset

resolutions = {
    'low': (128, 128, 64),
    'medium': (224, 224, 128),
    'high': (256, 256, 256)
}

for res_name, target_size in resolutions.items():
    print(f"\nCreating {res_name} resolution dataset...")
    
    resampling_dataset(
        nii_folder="source_data/",
        output_path=f"multi_res/{res_name}/",
        desired_volume=target_size,
        saving_mode="folder",
        debug=True
    )

print("\nMulti-resolution datasets created")
```

### Integration with Training Pipeline
Resample as preprocessing step:

```python
from nidataset.preprocessing import resampling_dataset
from nidataset.slices import extract_slices_dataset

# Step 1: Resample to uniform size
print("Step 1: Resampling volumes...")
resampling_dataset(
    nii_folder="training/raw_volumes/",
    output_path="training/uniform_volumes/",
    desired_volume=(224, 224, 128),
    saving_mode="folder",
    debug=True
)

# Step 2: Extract 2D slices from resampled volumes
print("\nStep 2: Extracting slices...")
extract_slices_dataset(
    nii_folder="training/uniform_volumes/",
    output_path="training/slices/",
    view="axial",
    saving_mode="case",
    target_size=(224, 224),
    save_stats=True
)

print("\nTraining data prepared")
```

### Memory-Efficient Processing
Process large datasets by resolution reduction:

```python
from nidataset.preprocessing import resampling_dataset
import nibabel as nib

# Calculate expected memory usage
def estimate_memory(shape, dtype='float32'):
    bytes_per_voxel = np.dtype(dtype).itemsize
    total_bytes = np.prod(shape) * bytes_per_voxel
    return total_bytes / (1024**3)  # GB

# Original sizes
print("Analyzing dataset...")
original_folder = "large_dataset/"
total_memory = 0

for filename in os.listdir(original_folder):
    if filename.endswith('.nii.gz'):
        img = nib.load(f"{original_folder}/{filename}")
        memory = estimate_memory(img.shape)
        total_memory += memory

print(f"Original dataset memory: {total_memory:.2f} GB")

# Resample to reduce size
target_size = (128, 128, 128)
reduced_memory = estimate_memory(target_size) * len(os.listdir(original_folder))
print(f"After resampling to {target_size}: {reduced_memory:.2f} GB")
print(f"Memory reduction: {(1 - reduced_memory/total_memory)*100:.1f}%")

# Perform resampling
resampling_dataset(
    nii_folder=original_folder,
    output_path="large_dataset_reduced/",
    desired_volume=target_size,
    saving_mode="folder",
    debug=True
)
```

### Batch Processing Different Datasets
Resample multiple datasets with different parameters:

```python
from nidataset.preprocessing import resampling_dataset

datasets = {
    'brain_mri': {'folder': 'data/brain/', 'size': (256, 256, 256)},
    'chest_ct': {'folder': 'data/chest/', 'size': (224, 224, 128)},
    'abdominal': {'folder': 'data/abdomen/', 'size': (224, 224, 128)}
}

for name, config in datasets.items():
    print(f"\nProcessing {name}...")
    
    resampling_dataset(
        nii_folder=config['folder'],
        output_path=f"resampled/{name}/",
        desired_volume=config['size'],
        saving_mode="case",
        debug=True
    )

print("\nAll datasets resampled")
```

### Creating Test/Train Splits
Resample and organize for machine learning:

```python
from nidataset.preprocessing import resampling_dataset
import shutil
import os

# Resample all data
resampling_dataset(
    nii_folder="all_data/",
    output_path="resampled_all/",
    desired_volume=(224, 224, 128),
    saving_mode="folder",
    debug=True
)

# Split into train/val/test
splits = {
    'train': ['scan_001', 'scan_002', 'scan_003'],
    'val': ['scan_004'],
    'test': ['scan_005']
}

for split_name, scan_ids in splits.items():
    split_folder = f"dataset/{split_name}/"
    os.makedirs(split_folder, exist_ok=True)
    
    for scan_id in scan_ids:
        src = f"resampled_all/{scan_id}_resampled.nii.gz"
        dst = f"{split_folder}/{scan_id}_resampled.nii.gz"
        if os.path.exists(src):
            shutil.copy(src, dst)
    
    print(f"{split_name}: {len(scan_ids)} scans")
```

### Validating Resampling Quality
Check interpolation quality and artifacts:

```python
import nibabel as nib
import numpy as np
from nidataset.preprocessing import resampling_dataset

# Resample
resampling_dataset(
    nii_folder="validation/original/",
    output_path="validation/resampled/",
    desired_volume=(224, 224, 128),
    saving_mode="folder",
    debug=True
)

# Validate quality
original = nib.load("validation/original/sample.nii.gz")
resampled = nib.load("validation/resampled/sample_resampled.nii.gz")

orig_data = original.get_fdata()
resamp_data = resampled.get_fdata()

# Check for artifacts
print("\nQuality Validation:")

# 1. Intensity range preservation
print(f"  Original range: [{orig_data.min():.1f}, {orig_data.max():.1f}]")
print(f"  Resampled range: [{resamp_data.min():.1f}, {resamp_data.max():.1f}]")

# 2. Mean intensity (should be similar)
orig_mean = orig_data[orig_data > 0].mean()
resamp_mean = resamp_data[resamp_data > 0].mean()
print(f"  Original mean: {orig_mean:.2f}")
print(f"  Resampled mean: {resamp_mean:.2f}")
print(f"  Difference: {abs(orig_mean - resamp_mean):.2f}")

# 3. Check for ringing artifacts (negative values in positive images)
if orig_data.min() >= 0 and resamp_data.min() < 0:
    print(f"  ⚠️ Warning: Negative values introduced (ringing artifact)")
else:
    print(f"  ✓ No negative values")
```

## Typical Workflow

```python
from nidataset.preprocessing import resampling_dataset
import nibabel as nib

# 1. Define input and output
source_folder = "dataset/variable_size_scans/"
output_folder = "dataset/uniform_size/"
target_dimensions = (224, 224, 128)

# 2. Resample all volumes to uniform size
resampling_dataset(
    nii_folder=source_folder,
    output_path=output_folder,
    desired_volume=target_dimensions,
    saving_mode="case",
    debug=True
)

# 3. Verify a sample result
sample = nib.load(f"{output_folder}/sample/sample_resampled.nii.gz")
print(f"\nResampled volume shape: {sample.shape}")
print(f"Target shape: {target_dimensions}")
print(f"Match: {sample.shape == target_dimensions}")

# 4. Use resampled volumes for:
# - Training deep learning models with fixed input size
# - Batch processing without dimension mismatches
# - Consistent preprocessing across datasets
# - Efficient memory usage
```