---
title: skull_CTA_dataset
parent: Package Functions
nav_order: 26
---

# `skull_CTA_dataset`

Batch process all angiography volumes in a dataset folder using a specialized skull-stripping pipeline optimized for contrast-enhanced imaging.

```python
skull_CTA_dataset(
    nii_folder: str,
    output_path: str,
    f_value: float = 0.1,
    clip_value: tuple = (0, 200),
    cleanup: bool = False,
    saving_mode: str = "case",
    debug: bool = False
) -> None
```

## Overview

This function applies a specialized skull-stripping pipeline to all angiography volumes in a dataset folder. The pipeline is specifically designed for contrast-enhanced imaging and includes intensity thresholding, Gaussian smoothing, FSL BET-based skull removal, and intensity clipping to isolate brain tissue and vessels.

**Processing pipeline** (applied to each volume):
1. **Intensity thresholding**: Remove background and extreme values
2. **Gaussian smoothing**: Reduce noise while preserving structures
3. **BET skull-stripping**: Remove skull using FSL's Brain Extraction Tool
4. **Intensity clipping**: Normalize intensity range for analysis

This is essential for:
- Removing skull and non-brain tissue from angiography scans
- Isolating brain vasculature for vessel analysis
- Standardizing intensity ranges across datasets
- Preparing data for vessel segmentation or detection
- Reducing false positives from bone in vessel detection

## Parameters

| Name          | Type    | Default      | Description                                                                                          |
|---------------|---------|--------------|------------------------------------------------------------------------------------------------------|
| `nii_folder`  | `str`   | *required*   | Path to the directory containing input angiography volumes in `.nii.gz` format.                     |
| `output_path` | `str`   | *required*   | Root directory where processed volumes will be saved.                                               |
| `f_value`     | `float` | `0.1`        | Fractional intensity threshold for BET (0-1). Lower values = more aggressive stripping.             |
| `clip_value`  | `tuple` | `(0, 200)`   | Intensity range `(min, max)` for final clipping after skull removal.                                |
| `cleanup`     | `bool`  | `False`      | If `True`, deletes intermediate files. In folder mode, also deletes temporary directories.          |
| `saving_mode` | `str`   | `"case"`     | Organization mode: `"case"` (folder per volume) or `"folder"` (separate skulled/masks folders).    |
| `debug`       | `bool`  | `False`      | If `True`, prints detailed processing information for each volume.                                  |

## Returns

`None` — The function saves processed volumes to disk.

## Output Organization

### Saving Modes

The function supports two organizational strategies for output files:

#### Case Mode (`saving_mode="case"`)
Creates a separate folder for each volume (recommended for dataset organization):

**Without cleanup** (`cleanup=False`):
```
output_path/
├── patient_001/
│   ├── patient_001_th.nii.gz  ← intermediate kept
│   ├── patient_001_th_sm.nii.gz  ← intermediate kept
│   ├── patient_001_th_sm_th.nii.gz  ← intermediate kept
│   ├── patient_001.skulled.nii.gz  ← intermediate kept
│   ├── patient_001.skulled.mask.nii.gz
│   └── patient_001.skulled.clipped.nii.gz
├── patient_002/
│   ├── patient_002_th.nii.gz  ← intermediate kept
│   ├── patient_002_th_sm.nii.gz  ← intermediate kept
│   ├── patient_002_th_sm_th.nii.gz  ← intermediate kept
│   ├── patient_002.skulled.nii.gz  ← intermediate kept
│   ├── patient_002.skulled.mask.nii.gz
│   └── patient_002.skulled.clipped.nii.gz
```

**With cleanup** (`cleanup=True`):
```
output_path/
├── patient_001/
│   ├── patient_001.skulled.mask.nii.gz
│   └── patient_001.skulled.clipped.nii.gz
├── patient_002/
│   ├── patient_002.skulled.mask.nii.gz
│   └── patient_002.skulled.clipped.nii.gz
```
*Note: Intermediate files (_th.nii.gz, _th_sm.nii.gz, _th_sm_th.nii.gz, .skulled.nii.gz) are deleted*

#### Folder Mode (`saving_mode="folder"`)
Separates skull-stripped images and masks into dedicated subdirectories. This mode uses temporary directories during processing.

**Processing flow**:
1. For each volume, creates a temporary directory: `output_path/_temp_<PREFIX>/`
2. Runs skull-stripping, generating files in the temporary directory
3. Moves `<PREFIX>.skulled.clipped.nii.gz` to `output_path/skulled/`
4. Moves `<PREFIX>.skulled.mask.nii.gz` to `output_path/masks/`
5. If `cleanup=True`, deletes the temporary directory
6. If `cleanup=False`, keeps the temporary directory with intermediate files

**Without cleanup** (`cleanup=False`):
```
output_path/
├── _temp_patient_001/
│   ├── patient_001_th.nii.gz  ← intermediate kept in temp
│   ├── patient_001_th_sm.nii.gz  ← intermediate kept in temp
│   ├── patient_001_th_sm_th.nii.gz  ← intermediate kept in temp
│   └── patient_001.skulled.nii.gz  ← intermediate kept in temp
├── _temp_patient_002/
│   ├── patient_002_th.nii.gz  ← intermediate kept in temp
│   ├── patient_002_th_sm.nii.gz  ← intermediate kept in temp
│   ├── patient_002_th_sm_th.nii.gz  ← intermediate kept in temp
│   └── patient_002.skulled.nii.gz  ← intermediate kept in temp
├── skulled/
│   ├── patient_001.skulled.clipped.nii.gz
│   ├── patient_002.skulled.clipped.nii.gz
│   └── patient_003.skulled.clipped.nii.gz
└── masks/
    ├── patient_001.skulled.mask.nii.gz
    ├── patient_002.skulled.mask.nii.gz
    └── patient_003.skulled.mask.nii.gz
```

**With cleanup** (`cleanup=True`):
```
output_path/
├── skulled/
│   ├── patient_001.skulled.clipped.nii.gz
│   ├── patient_002.skulled.clipped.nii.gz
│   └── patient_003.skulled.clipped.nii.gz
└── masks/
    ├── patient_001.skulled.mask.nii.gz
    ├── patient_002.skulled.mask.nii.gz
    └── patient_003.skulled.mask.nii.gz
```

**Important notes for folder mode**:
- Temporary directories (`_temp_<PREFIX>/`) are created for each volume during processing
- If `cleanup=True`, temporary directories are deleted after moving final outputs
- If `cleanup=False`, temporary directories are kept with their intermediate files
- This allows you to inspect preprocessing steps or recover intermediate volumes if needed

## Output Files

The function generates multiple files during processing:

### Intermediate Files (removed if cleanup=True)

| File                           | Description                                       |
|--------------------------------|---------------------------------------------------|
| `<PREFIX>_th.nii.gz`           | After initial thresholding [0-100]               |
| `<PREFIX>_th_sm.nii.gz`        | After Gaussian smoothing (sigma=1)               |
| `<PREFIX>_th_sm_th.nii.gz`     | After secondary thresholding [0-100]             |
| `<PREFIX>.skulled.nii.gz`      | BET output before final clipping                 |

### Final Output Files (always retained)

| File                                  | Description                                       |
|---------------------------------------|---------------------------------------------------|
| `<PREFIX>.skulled.mask.nii.gz`        | Binary brain mask from BET                        |
| `<PREFIX>.skulled.clipped.nii.gz`     | Final skull-stripped and intensity-clipped volume |

**Example**: Input `scan_042.nii.gz` produces:
- `scan_042.skulled.mask.nii.gz`
- `scan_042.skulled.clipped.nii.gz`

## BET Parameters

### f_value (Fractional Intensity Threshold)

The `f_value` parameter controls the aggressiveness of skull removal:

| f_value | Effect                           | Use Case                              |
|---------|----------------------------------|---------------------------------------|
| 0.05    | Very aggressive, may remove brain| Thick skull, heavy calcification      |
| 0.1     | Aggressive stripping (default)   | Standard angiography scans            |
| 0.2     | Moderate stripping               | Thin skull, pediatric scans           |
| 0.3-0.5 | Conservative stripping           | Preserve all brain tissue             |

**Selection guidelines**:
- **Lower values (0.05-0.15)**: More skull removed, risk of removing brain
- **Higher values (0.2-0.5)**: Less aggressive, may leave some skull
- **Default 0.1**: Good balance for most angiography applications

### clip_value (Intensity Range)

The `clip_value` tuple defines the final intensity range:

| Range       | Use Case                                    |
|-------------|---------------------------------------------|
| (0, 100)    | Conservative, soft tissue only              |
| (0, 200)    | Standard for angiography (default)          |
| (0, 300)    | Include high-intensity contrast regions     |
| (0, 500)    | Very inclusive, may include artifacts       |

## FSL Requirements

**Important**: This function requires FSL (FMRIB Software Library) to be installed and accessible:

1. **Installation**: Download and install FSL from https://fsl.fmrib.ox.ac.uk/
2. **Environment**: FSL environment variables must be configured
3. **Command-line access**: Tools `fslmaths` and `bet` must be in PATH
4. **Execution**: Run scripts from terminal, not IDE (to ensure environment detection)

**Verify FSL installation**:
```bash
which bet  # Should return path to bet executable
bet  # Should show BET help message
```

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |
| `ValueError`         | Invalid `saving_mode` parameter                                    |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **Pre-centered Volumes**: Input scans should be centered on brain region
- **No FOV Cropping**: Function preserves original spatial dimensions
- **FSL Dependency**: FSL must be properly installed and configured
- **Sequential Processing**: Volumes processed one at a time with progress bar (tqdm)
- **Output Directories**: Automatically created if they don't exist
- **Temporary Directories**: In folder mode, `_temp_` directories are used during processing

## Examples

### Basic Usage - Case Mode
Process dataset with default parameters:

```python
from nidataset.preprocessing import skull_CTA_dataset

skull_CTA_dataset(
    nii_folder="dataset/angiography/",
    output_path="dataset/skull_stripped/",
    f_value=0.1,
    clip_value=(0, 200),
    saving_mode="case"
)
# Creates: dataset/skull_stripped/scan_001/scan_001.skulled.clipped.nii.gz, ...
```

### Folder Mode Organization
Separate skull-stripped images and masks:

```python
skull_CTA_dataset(
    nii_folder="cta_volumes/",
    output_path="processed/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=True,
    saving_mode="folder",
    debug=True
)
# Images in processed/skulled/, masks in processed/masks/
```

### With Cleanup
Remove intermediate files to save space:

```python
skull_CTA_dataset(
    nii_folder="data/cta_scans/",
    output_path="data/processed/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=True,  # Remove intermediates
    saving_mode="case",
    debug=True
)
# Only masks and final skull-stripped volumes retained
```

### Preserving Intermediates in Folder Mode
Keep temporary directories for inspection:

```python
skull_CTA_dataset(
    nii_folder="data/cta_scans/",
    output_path="data/processed/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=False,  # Keep temp directories with intermediates
    saving_mode="folder",
    debug=True
)
# Final outputs in skulled/ and masks/, intermediates in _temp_*/ directories
```

### Aggressive Skull Removal
Use lower f_value for thick skull or heavy calcification:

```python
skull_CTA_dataset(
    nii_folder="thick_skull_cases/",
    output_path="aggressive_strip/",
    f_value=0.05,  # More aggressive
    clip_value=(0, 200),
    saving_mode="case",
    debug=True
)
```

### Conservative Stripping
Use higher f_value to preserve all brain tissue:

```python
skull_CTA_dataset(
    nii_folder="pediatric_scans/",
    output_path="conservative_strip/",
    f_value=0.3,  # Less aggressive
    clip_value=(0, 200),
    saving_mode="case",
    debug=True
)
```

### Custom Intensity Range
Adjust clipping for different contrast levels:

```python
skull_CTA_dataset(
    nii_folder="high_contrast_cta/",
    output_path="processed/",
    f_value=0.1,
    clip_value=(0, 300),  # Include higher intensities
    cleanup=True,
    saving_mode="case",
    debug=True
)
```

### Quality Control Workflow
Process and verify skull removal:

```python
import nibabel as nib
import numpy as np
from nidataset.preprocessing import skull_CTA_dataset

# Process dataset
skull_CTA_dataset(
    nii_folder="qa/original/",
    output_path="qa/processed/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=False,  # Keep intermediates for inspection
    saving_mode="case",
    debug=True
)

# Verify a sample result
original = nib.load("qa/original/sample.nii.gz")
mask = nib.load("qa/processed/sample/sample.skulled.mask.nii.gz")
stripped = nib.load("qa/processed/sample/sample.skulled.clipped.nii.gz")

orig_data = original.get_fdata()
mask_data = mask.get_fdata()
stripped_data = stripped.get_fdata()

print("\nQuality Control:")
print(f"  Original shape: {orig_data.shape}")
print(f"  Mask shape: {mask_data.shape}")
print(f"  Stripped shape: {stripped_data.shape}")

# Calculate brain coverage
brain_voxels = np.sum(mask_data > 0)
total_voxels = np.prod(mask_data.shape)
coverage = (brain_voxels / total_voxels) * 100

print(f"\n  Brain coverage: {coverage:.1f}%")
print(f"  Expected: 20-40% for head scans")

# Check intensity range
print(f"\n  Original intensity: [{orig_data.min():.1f}, {orig_data.max():.1f}]")
print(f"  Stripped intensity: [{stripped_data.min():.1f}, {stripped_data.max():.1f}]")

if coverage < 15 or coverage > 50:
    print(f"  ⚠️ Warning: Unusual coverage, check f_value setting")
```

### Multi-Center Dataset Processing
Process scans from different sites:

```python
from nidataset.preprocessing import skull_CTA_dataset

centers = {
    'center_A': {'f_value': 0.1, 'clip': (0, 200)},
    'center_B': {'f_value': 0.15, 'clip': (0, 250)},  # Different protocol
    'center_C': {'f_value': 0.1, 'clip': (0, 200)}
}

for center, params in centers.items():
    print(f"\nProcessing {center}...")
    
    skull_CTA_dataset(
        nii_folder=f"data/{center}/raw/",
        output_path=f"data/{center}/processed/",
        f_value=params['f_value'],
        clip_value=params['clip'],
        cleanup=True,
        saving_mode="case",
        debug=True
    )

print("\nAll centers processed")
```

### Using Masks for Registration
Apply masks to registration workflow:

```python
from nidataset.preprocessing import skull_CTA_dataset, register_CTA_dataset
import os
import shutil

# Step 1: Skull-strip all volumes to generate masks
print("Step 1: Skull-stripping to generate masks...")
skull_CTA_dataset(
    nii_folder="raw_scans/",
    output_path="skull_stripped/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=True,
    saving_mode="folder",
    debug=True
)

# Step 2: Register using generated masks
print("\nStep 2: Registering with skull-stripped masks...")
register_CTA_dataset(
    nii_folder="raw_scans/",
    mask_folder="skull_stripped/masks/",
    template_path="template/template.nii.gz",
    template_mask_path="template/template_mask.nii.gz",
    output_path="registered/",
    saving_mode="case",
    cleanup=True,
    debug=True
)

print("\nPipeline complete")
```

## Typical Workflow

```python
from nidataset.preprocessing import skull_CTA_dataset
import nibabel as nib
import numpy as np

# 1. Verify FSL is installed
import subprocess
try:
    subprocess.run(['bet', '--version'], capture_output=True, check=True)
    print("FSL is properly configured")
except:
    print("Error: FSL not found. Please install and configure FSL")
    exit(1)

# 2. Process all angiography scans
skull_CTA_dataset(
    nii_folder="data/angiography_scans/",
    output_path="data/skull_stripped/",
    f_value=0.1,  # Adjust based on dataset
    clip_value=(0, 200),
    cleanup=True,  # Save disk space
    saving_mode="case",
    debug=True
)

# 3. Verify a sample result
mask = nib.load("data/skull_stripped/sample/sample.skulled.mask.nii.gz")
stripped = nib.load("data/skull_stripped/sample/sample.skulled.clipped.nii.gz")

mask_data = mask.get_fdata()
coverage = np.sum(mask_data > 0) / np.prod(mask_data.shape) * 100

print(f"\nQuality check:")
print(f"  Brain coverage: {coverage:.1f}%")
print(f"  Stripped shape: {stripped.shape}")

# 4. Use skull-stripped volumes for:
# - Vessel segmentation
# - Aneurysm detection
# - Vascular analysis
# - Registration without skull influence
```