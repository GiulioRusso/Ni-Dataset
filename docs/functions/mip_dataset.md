---
title: mip_dataset
parent: Package Functions
nav_order: 16
---

# `mip_dataset`

Batch process all volumes in a dataset folder to generate sliding-window Maximum Intensity Projections along a specified anatomical plane.

```python
mip_dataset(
    nii_folder: str,
    output_path: str,
    window_size: int = 10,
    view: str = "axial",
    saving_mode: str = "case",
    debug: bool = False
) -> None
```

## Overview

This function processes all NIfTI volumes in a dataset folder by applying sliding-window Maximum Intensity Projection to each file. For every slice position, it computes the maximum intensity across a local neighborhood, creating an enhanced volume that highlights high-intensity structures like blood vessels or contrast-enhanced regions.

The MIP operation is useful for:
- Enhancing vascular structures in angiography scans
- Improving visibility of contrast-enhanced regions
- Reducing noise while preserving peak intensities
- Creating enhanced datasets for vessel detection or segmentation

Each output volume maintains the same dimensions and spatial metadata as its input, with filenames following the pattern `<PREFIX>_mip_<VIEW>.nii.gz`.

## Parameters

| Name          | Type   | Default    | Description                                                                                          |
|---------------|--------|------------|------------------------------------------------------------------------------------------------------|
| `nii_folder`  | `str`  | *required* | Path to the directory containing NIfTI volumes in `.nii.gz` format.                                 |
| `output_path` | `str`  | *required* | Root directory where MIP volumes will be saved.                                                     |
| `window_size` | `int`  | `10`       | Number of slices on each side of the current position for projection. Effective window: `2×size+1`. |
| `view`        | `str`  | `"axial"`  | Anatomical plane for projection: `"axial"`, `"coronal"`, or `"sagittal"`.                          |
| `saving_mode` | `str`  | `"case"`   | Organization mode: `"case"` (folder per file) or `"view"` (shared folder).                         |
| `debug`       | `bool` | `False`    | If `True`, prints processing summary after completion.                                              |

## Returns

`None` – The function saves MIP volumes to disk.

## Output Organization

### Saving Modes

#### Case Mode (`saving_mode="case"`)
Creates a separate folder for each input volume:
```
output_path/
├── patient_001/
│   └── axial/
│       └── patient_001_mip_axial.nii.gz
├── patient_002/
│   └── axial/
│       └── patient_002_mip_axial.nii.gz
```

#### View Mode (`saving_mode="view"`)
Groups all MIP volumes in a single view folder:
```
output_path/
└── axial/
    ├── patient_001_mip_axial.nii.gz
    ├── patient_002_mip_axial.nii.gz
    └── ...
```

### Filename Pattern
Each MIP volume is saved as:
```
<PREFIX>_mip_<VIEW>.nii.gz
```

**Example**: Input `scan_042.nii.gz` → Output `scan_042_mip_axial.nii.gz`

## Window Size Parameter

The `window_size` parameter controls the neighborhood size for maximum intensity computation:

**Effective window length**: `2 × window_size + 1`

| Window Size | Effective Slices | Effect                                  | Best For                    |
|-------------|------------------|-----------------------------------------|-----------------------------|
| 5           | 11 slices        | Minimal smoothing, local enhancement    | Fine vessel details         |
| 10          | 21 slices        | Moderate enhancement, balanced          | General purpose (default)   |
| 20          | 41 slices        | Strong enhancement, wider context       | Large vessel structures     |
| 30+         | 61+ slices       | Maximum enhancement, global view        | Overview visualization      |

**Trade-off**: Larger windows enhance more structures but may blur fine details and mix unrelated regions.

## Anatomical Views

The `view` parameter determines the projection axis:

| View         | Projection Axis | Description                    | Common Applications          |
|--------------|-----------------|--------------------------------|------------------------------|
| `"axial"`    | Z-axis          | Top-down projection            | Brain vessels, chest imaging |
| `"coronal"`  | Y-axis          | Front-back projection          | Spine vessels, full body     |
| `"sagittal"` | X-axis          | Left-right projection          | Bilateral vessel comparison  |

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |
| `ValueError`         | Invalid `view` or `saving_mode` parameter                          |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **3D Volumes Required**: Input must be 3D NIfTI images
- **Sequential Processing**: Files are processed one at a time (no parallelization)
- **Progress Display**: Shows progress bar with current file being processed
- **Output Directory**: Automatically created if it doesn't exist
- **Spatial Metadata**: Affine transformations are preserved from input

## Examples

### Basic Usage
Generate MIP volumes with default settings:

```python
from nidataset.preprocessing import mip_dataset

mip_dataset(
    nii_folder="dataset/angiography/",
    output_path="dataset/mip_enhanced/",
    window_size=10,
    view="axial",
    saving_mode="case"
)
# Creates: dataset/mip_enhanced/case_001/axial/case_001_mip_axial.nii.gz, ...
```

### Large Window for Strong Enhancement
Use larger window for prominent vessel visualization:

```python
mip_dataset(
    nii_folder="data/vessel_scans/",
    output_path="data/enhanced_vessels/",
    window_size=20,  # 41-slice window
    view="axial",
    saving_mode="case",
    debug=True
)
# Prints summary after processing
```

### Small Window for Detail Preservation
Use smaller window to preserve fine structures:

```python
mip_dataset(
    nii_folder="high_res_scans/",
    output_path="detail_preserved_mip/",
    window_size=5,  # 11-slice window
    view="coronal",
    saving_mode="view",
    debug=True
)
```

### Multi-View Processing
Generate MIP volumes for all anatomical views:

```python
from nidataset.preprocessing import mip_dataset

views = ["axial", "coronal", "sagittal"]
base_path = "mip_multi_view/"

for view in views:
    print(f"Processing {view} view...")
    mip_dataset(
        nii_folder="original_scans/",
        output_path=base_path,
        window_size=15,
        view=view,
        saving_mode="view",
        debug=True
    )
# Creates separate folders for each view
```

### Processing Different Datasets
Apply MIP with optimized parameters for different scan types:

```python
from nidataset.preprocessing import mip_dataset

datasets = {
    'head_cta': {'folder': 'data/head_cta/', 'window': 10, 'view': 'axial'},
    'neck_cta': {'folder': 'data/neck_cta/', 'window': 15, 'view': 'coronal'},
    'chest_cta': {'folder': 'data/chest_cta/', 'window': 20, 'view': 'axial'}
}

for name, config in datasets.items():
    print(f"Processing {name}...")
    mip_dataset(
        nii_folder=config['folder'],
        output_path=f"mip_results/{name}/",
        window_size=config['window'],
        view=config['view'],
        saving_mode="case",
        debug=True
    )
```

### Comparing Window Sizes
Test different window sizes to find optimal enhancement:

```python
from nidataset.preprocessing import mip_dataset

window_sizes = [5, 10, 15, 20, 30]
test_folder = "test_scans/"

for size in window_sizes:
    mip_dataset(
        nii_folder=test_folder,
        output_path=f"window_comparison/size_{size}/",
        window_size=size,
        view="axial",
        saving_mode="view",
        debug=True
    )
    print(f"Completed window size {size} (effective: {2*size+1} slices)")

print("\nCompare outputs visually to select optimal window size")
```

### Quality Control Workflow
Generate MIP and verify enhancement quality:

```python
import nibabel as nib
import numpy as np
from nidataset.preprocessing import mip_dataset

# Generate MIP volumes
mip_dataset(
    nii_folder="qa/originals/",
    output_path="qa/mip_processed/",
    window_size=15,
    view="axial",
    saving_mode="case",
    debug=True
)

# Load and compare original vs MIP
original = nib.load("qa/originals/sample.nii.gz")
mip_vol = nib.load("qa/mip_processed/sample/axial/sample_mip_axial.nii.gz")

orig_data = original.get_fdata()
mip_data = mip_vol.get_fdata()

print(f"\nQuality Control:")
print(f"  Original shape: {orig_data.shape}")
print(f"  MIP shape: {mip_data.shape}")
print(f"  Original intensity range: [{orig_data.min():.1f}, {orig_data.max():.1f}]")
print(f"  MIP intensity range: [{mip_data.min():.1f}, {mip_data.max():.1f}]")

# Check enhancement (MIP should have higher mean in vessel regions)
vessel_region = orig_data[100:150, 100:150, :]  # Example region
vessel_mip = mip_data[100:150, 100:150, :]

print(f"  Original vessel region mean: {vessel_region.mean():.1f}")
print(f"  MIP vessel region mean: {vessel_mip.mean():.1f}")
print(f"  Enhancement factor: {vessel_mip.mean() / vessel_region.mean():.2f}x")
```

### Integration with Segmentation Pipeline
Use MIP as preprocessing for vessel segmentation:

```python
from nidataset.preprocessing import mip_dataset
from nidataset.volume import generate_brain_mask

# Step 1: Generate MIP volumes to enhance vessels
mip_dataset(
    nii_folder="pipeline/raw_scans/",
    output_path="pipeline/mip_enhanced/",
    window_size=15,
    view="axial",
    saving_mode="case",
    debug=True
)

# Step 2: Generate brain masks for MIP volumes
generate_brain_mask_dataset(
    nii_folder="pipeline/mip_enhanced/",
    output_path="pipeline/brain_masks/",
    threshold=(80, 400),  # Higher threshold for enhanced vessels
    closing_radius=3,
    debug=True
)

# Step 3: Apply masks to MIP volumes
import nibabel as nib
import os

mip_folder = "pipeline/mip_enhanced/"
mask_folder = "pipeline/brain_masks/"
output_folder = "pipeline/vessel_focused/"
os.makedirs(output_folder, exist_ok=True)

for case_folder in os.listdir(mip_folder):
    case_path = os.path.join(mip_folder, case_folder, "axial")
    if os.path.isdir(case_path):
        for mip_file in os.listdir(case_path):
            if mip_file.endswith('_mip_axial.nii.gz'):
                # Load MIP and mask
                mip = nib.load(os.path.join(case_path, mip_file))
                mip_data = mip.get_fdata()
                
                mask_file = mip_file.replace('_mip_axial', '_brain_mask')
                mask = nib.load(os.path.join(mask_folder, mask_file))
                mask_data = mask.get_fdata()
                
                # Apply mask
                vessel_focused = mip_data * mask_data
                
                # Save
                output_img = nib.Nifti1Image(vessel_focused, mip.affine)
                output_path = os.path.join(output_folder, mip_file)
                nib.save(output_img, output_path)

print("Vessel-focused volumes ready for segmentation")
```

### Batch Processing with Progress Tracking
Process large datasets with detailed progress:

```python
from nidataset.preprocessing import mip_dataset
import time

datasets = ['dataset_A', 'dataset_B', 'dataset_C']
total_start = time.time()

for i, dataset in enumerate(datasets, 1):
    print(f"\n{'='*50}")
    print(f"Processing dataset {i}/{len(datasets)}: {dataset}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    mip_dataset(
        nii_folder=f"data/{dataset}/",
        output_path=f"results/{dataset}_mip/",
        window_size=15,
        view="axial",
        saving_mode="case",
        debug=True
    )
    
    elapsed = time.time() - start_time
    print(f"Completed {dataset} in {elapsed:.1f} seconds")

total_elapsed = time.time() - total_start
print(f"\n{'='*50}")
print(f"All datasets processed in {total_elapsed:.1f} seconds")
```

### Creating Training Data with MIP Enhancement
Prepare enhanced training dataset:

```python
from nidataset.preprocessing import mip_dataset
from nidataset.slices import extract_slices_dataset

# Step 1: Generate MIP-enhanced volumes
mip_dataset(
    nii_folder="training/raw_volumes/",
    output_path="training/mip_volumes/",
    window_size=15,
    view="axial",
    saving_mode="case",
    debug=True
)

# Step 2: Extract 2D slices from MIP volumes
# First, collect all MIP files
import os
mip_files = []
for case in os.listdir("training/mip_volumes/"):
    case_path = os.path.join("training/mip_volumes/", case, "axial")
    if os.path.isdir(case_path):
        for f in os.listdir(case_path):
            if f.endswith('.nii.gz'):
                mip_files.append(os.path.join(case_path, f))

# Create temporary folder with MIP files
temp_mip = "training/temp_mip/"
os.makedirs(temp_mip, exist_ok=True)
for mip_file in mip_files:
    shutil.copy(mip_file, temp_mip)

# Extract slices
extract_slices_dataset(
    nii_folder=temp_mip,
    output_path="training/mip_slices/",
    view="axial",
    saving_mode="case",
    target_size=(512, 512),
    save_stats=True
)

print("MIP-enhanced training data prepared")
```

### Analyzing Enhancement Effect
Quantify the MIP enhancement across dataset:

```python
import nibabel as nib
import numpy as np
from nidataset.preprocessing import mip_dataset
import pandas as pd

# Generate MIP volumes
mip_dataset(
    nii_folder="analysis/originals/",
    output_path="analysis/mip/",
    window_size=15,
    view="axial",
    saving_mode="view",
    debug=True
)

# Analyze enhancement
results = []
for orig_file in os.listdir("analysis/originals/"):
    if orig_file.endswith('.nii.gz'):
        # Load original and MIP
        orig = nib.load(f"analysis/originals/{orig_file}")
        orig_data = orig.get_fdata()
        
        mip_file = orig_file.replace('.nii.gz', '_mip_axial.nii.gz')
        mip = nib.load(f"analysis/mip/axial/{mip_file}")
        mip_data = mip.get_fdata()
        
        # Calculate metrics
        orig_mean = orig_data[orig_data > 0].mean()
        mip_mean = mip_data[mip_data > 0].mean()
        
        orig_max = orig_data.max()
        mip_max = mip_data.max()
        
        enhancement_ratio = mip_mean / orig_mean
        
        results.append({
            'file': orig_file,
            'orig_mean': orig_mean,
            'mip_mean': mip_mean,
            'enhancement': enhancement_ratio,
            'orig_max': orig_max,
            'mip_max': mip_max
        })

# Create summary
df = pd.DataFrame(results)
print("\nMIP Enhancement Analysis:")
print(f"  Average enhancement: {df['enhancement'].mean():.2f}x")
print(f"  Min enhancement: {df['enhancement'].min():.2f}x")
print(f"  Max enhancement: {df['enhancement'].max():.2f}x")
print(f"\n  Mean intensity increase: {(df['mip_mean'] - df['orig_mean']).mean():.1f}")
```

## Typical Workflow

```python
from nidataset.preprocessing import mip_dataset
import nibabel as nib

# 1. Define input and output paths
scan_folder = "dataset/angiography_scans/"
mip_output = "dataset/mip_enhanced/"

# 2. Generate MIP volumes with appropriate window size
mip_dataset(
    nii_folder=scan_folder,
    output_path=mip_output,
    window_size=15,  # Adjust based on vessel size
    view="axial",
    saving_mode="case",
    debug=True
)

# 3. Verify a sample result
sample_orig = nib.load("dataset/angiography_scans/sample.nii.gz")
sample_mip = nib.load("dataset/mip_enhanced/sample/axial/sample_mip_axial.nii.gz")

print(f"Original shape: {sample_orig.shape}")
print(f"MIP shape: {sample_mip.shape}")

# 4. Use MIP volumes for:
# - Vessel detection and segmentation
# - Aneurysm identification
# - Vascular analysis
# - Training data for deep learning models
```