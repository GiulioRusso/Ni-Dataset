---
title: skull_CTA_dataset
parent: Package Functions
nav_order: 4
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
| `cleanup`     | `bool`  | `False`      | If `True`, deletes intermediate thresholded and smoothed images. Mask and final volume retained.    |
| `saving_mode` | `str`   | `"case"`     | Organization mode: `"case"` (folder per volume) or `"folder"` (shared folder).                     |
| `debug`       | `bool`  | `False`      | If `True`, prints detailed processing information for each volume.                                  |

## Returns

`None` – The function saves processed volumes to disk.

## Output Organization

### Saving Modes

#### Case Mode (`saving_mode="case"`)
Creates a separate folder for each volume (recommended):
```
output_path/
├── patient_001/
│   ├── patient_001_thresholded.nii.gz (if cleanup=False)
│   ├── patient_001_smoothed.nii.gz (if cleanup=False)
│   ├── patient_001_mask.nii.gz
│   └── patient_001_skull_stripped.nii.gz
├── patient_002/
│   └── ...
```

#### Folder Mode (`saving_mode="folder"`)
All processed volumes in a single directory:
```
output_path/
├── patient_001_thresholded.nii.gz (if cleanup=False)
├── patient_001_smoothed.nii.gz (if cleanup=False)
├── patient_001_mask.nii.gz
├── patient_001_skull_stripped.nii.gz
├── patient_002_mask.nii.gz
├── patient_002_skull_stripped.nii.gz
└── ...
```

### Output Files

For each input volume (when `cleanup=False`):

| File                              | Description                                       | Kept After Cleanup |
|-----------------------------------|---------------------------------------------------|--------------------|
| `<PREFIX>_thresholded.nii.gz`     | After initial intensity thresholding              | No                 |
| `<PREFIX>_smoothed.nii.gz`        | After Gaussian smoothing                          | No                 |
| `<PREFIX>_mask.nii.gz`            | Binary brain mask from BET                        | Yes                |
| `<PREFIX>_skull_stripped.nii.gz`  | Final skull-stripped and clipped volume           | Yes                |

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
- **Sequential Processing**: Volumes processed one at a time
- **Progress Display**: Shows progress bar during processing
- **Output Directories**: Automatically created if they don't exist

## Examples

### Basic Usage
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
# Creates: dataset/skull_stripped/scan_001/scan_001_skull_stripped.nii.gz, ...
```

### With Cleanup
Remove intermediate files to save space:

```python
skull_CTA_dataset(
    nii_folder="data/cta_scans/",
    output_path="data/processed/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=True,  # Remove thresholded and smoothed intermediates
    saving_mode="case",
    debug=True
)
# Only masks and final skull-stripped volumes retained
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

### Folder Mode Organization
All outputs in single directory:

```python
skull_CTA_dataset(
    nii_folder="cta_volumes/",
    output_path="all_stripped/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=True,
    saving_mode="folder",
    debug=True
)
# All processed files in one folder
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
mask = nib.load("qa/processed/sample/sample_mask.nii.gz")
stripped = nib.load("qa/processed/sample/sample_skull_stripped.nii.gz")

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

### Testing Different f_values
Find optimal skull-stripping threshold:

```python
from nidataset.preprocessing import skull_CTA_dataset

test_f_values = [0.05, 0.1, 0.15, 0.2, 0.3]

for f_val in test_f_values:
    print(f"\nTesting f_value = {f_val}")
    
    skull_CTA_dataset(
        nii_folder="test_scan/",
        output_path=f"f_value_test/f_{f_val}/",
        f_value=f_val,
        clip_value=(0, 200),
        cleanup=False,
        saving_mode="folder",
        debug=True
    )

print("\nCompare outputs visually to select optimal f_value")
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

### Integration with Vessel Analysis Pipeline
Complete preprocessing for vessel detection:

```python
from nidataset.preprocessing import skull_CTA_dataset, mip_dataset

# Step 1: Skull stripping
print("Step 1: Skull stripping...")
skull_CTA_dataset(
    nii_folder="pipeline/raw_cta/",
    output_path="pipeline/skull_stripped/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=True,
    saving_mode="case",
    debug=True
)

# Step 2: Collect skull-stripped files
import os
stripped_folder = "pipeline/collected_stripped/"
os.makedirs(stripped_folder, exist_ok=True)

for case in os.listdir("pipeline/skull_stripped/"):
    case_path = os.path.join("pipeline/skull_stripped/", case)
    if os.path.isdir(case_path):
        stripped_file = f"{case_path}/{case}_skull_stripped.nii.gz"
        if os.path.exists(stripped_file):
            shutil.copy(stripped_file, stripped_folder)

# Step 3: Apply MIP to enhance vessels
print("\nStep 2: Applying MIP enhancement...")
mip_dataset(
    nii_folder=stripped_folder,
    output_path="pipeline/vessel_enhanced/",
    window_size=15,
    view="axial",
    saving_mode="case",
    debug=True
)

print("\nPipeline complete: Ready for vessel segmentation")
```

### Batch Processing with Error Handling
Process large datasets robustly:

```python
from nidataset.preprocessing import skull_CTA_dataset
import os

# First, verify FSL is available
import subprocess
try:
    subprocess.run(['bet'], capture_output=True, check=False)
    print("✓ FSL/BET is available")
except FileNotFoundError:
    print("✗ Error: FSL/BET not found in PATH")
    print("Please install FSL and configure environment variables")
    exit(1)

# Process dataset
try:
    skull_CTA_dataset(
        nii_folder="large_dataset/",
        output_path="large_dataset_processed/",
        f_value=0.1,
        clip_value=(0, 200),
        cleanup=True,
        saving_mode="case",
        debug=True
    )
    print("\n✓ Processing completed successfully")
    
except FileNotFoundError as e:
    print(f"\n✗ Error: {e}")
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
```

### Comparing Stripping Methods
Evaluate different parameter combinations:

```python
from nidataset.preprocessing import skull_CTA_dataset
import nibabel as nib
import numpy as np
import pandas as pd

# Test different configurations
configs = [
    {'f': 0.05, 'clip': (0, 200), 'name': 'aggressive'},
    {'f': 0.1, 'clip': (0, 200), 'name': 'standard'},
    {'f': 0.2, 'clip': (0, 200), 'name': 'conservative'}
]

results = []

for config in configs:
    print(f"\nTesting {config['name']} configuration...")
    
    skull_CTA_dataset(
        nii_folder="comparison/original/",
        output_path=f"comparison/{config['name']}/",
        f_value=config['f'],
        clip_value=config['clip'],
        cleanup=False,
        saving_mode="folder",
        debug=True
    )
    
    # Analyze result
    mask = nib.load(f"comparison/{config['name']}/sample_mask.nii.gz")
    mask_data = mask.get_fdata()
    
    coverage = np.sum(mask_data > 0) / np.prod(mask_data.shape) * 100
    
    results.append({
        'configuration': config['name'],
        'f_value': config['f'],
        'brain_coverage': coverage
    })

# Display comparison
df = pd.DataFrame(results)
print("\nConfiguration Comparison:")
print(df.to_string(index=False))
```

### Creating Visualization for QC
Generate overlays for quality control:

```python
import nibabel as nib
import numpy as np
from nidataset.preprocessing import skull_CTA_dataset

# Process
skull_CTA_dataset(
    nii_folder="visualization/scans/",
    output_path="visualization/processed/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=False,
    saving_mode="case",
    debug=True
)

# Create overlay
original = nib.load("visualization/scans/sample.nii.gz")
mask = nib.load("visualization/processed/sample/sample_mask.nii.gz")

orig_data = original.get_fdata()
mask_data = mask.get_fdata()

# Create edge overlay
from scipy import ndimage
edges = ndimage.sobel(mask_data)

overlay = orig_data.copy()
overlay[edges > 0] = orig_data.max()  # Highlight mask edges

# Save overlay
overlay_img = nib.Nifti1Image(overlay, original.affine)
nib.save(overlay_img, "visualization/qc_overlay.nii.gz")
print("QC overlay created: visualization/qc_overlay.nii.gz")
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
mask = nib.load("data/skull_stripped/sample/sample_mask.nii.gz")
stripped = nib.load("data/skull_stripped/sample/sample_skull_stripped.nii.gz")

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