---
title: register_CTA_dataset
parent: Package Functions
nav_order: 10
---

# `register_CTA_dataset`

Batch register all medical imaging volumes in a dataset folder to a reference template using intensity-based registration with mutual information.

```python
register_CTA_dataset(
    nii_folder: str,
    mask_folder: str,
    template_path: str,
    template_mask_path: str,
    output_image_path: str,
    output_transformation_path: str = "",
    saving_mode: str = "case",
    cleanup: bool = False,
    debug: bool = False
) -> None
```

## Overview

This function processes all volumes in a dataset folder by aligning them to a common reference template through image registration. Each volume undergoes:

1. **Preprocessing**: Gaussian filtering to remove noise and outliers
2. **Mask application**: Brain region isolation using provided masks
3. **Registration**: Alignment to template using mutual information metric
4. **Transform saving**: Storing transformation parameters for later use

Registration is essential for:
- Standardizing spatial orientation across datasets
- Enabling voxel-wise analysis and comparisons
- Creating anatomically aligned datasets for machine learning
- Normalizing scan positions and orientations
- Building anatomical atlases

## Parameters

| Name                         | Type   | Default    | Description                                                                                          |
|------------------------------|--------|------------|------------------------------------------------------------------------------------------------------|
| `nii_folder`                 | `str`  | *required* | Path to the directory containing input volumes in `.nii.gz` format.                                 |
| `mask_folder`                | `str`  | *required* | Path to the directory containing brain masks. Must have matching filenames.                         |
| `template_path`              | `str`  | *required* | Path to the reference template volume for registration.                                             |
| `template_mask_path`         | `str`  | *required* | Path to the template's brain mask.                                                                  |
| `output_image_path`          | `str`  | *required* | Directory where registered volumes will be saved.                                                   |
| `output_transformation_path` | `str`  | `""`       | Directory for transformation files. Ignored in case mode (stored with images).                      |
| `saving_mode`                | `str`  | `"case"`   | Organization mode: `"case"` (folder per volume) or `"folder"` (shared folders).                    |
| `cleanup`                    | `bool` | `False`    | If `True`, deletes intermediate Gaussian-filtered files after registration.                         |
| `debug`                      | `bool` | `False`    | If `True`, prints detailed registration information for each volume.                                |

## Returns

`None` – The function saves registered volumes and transformations to disk.

## Output Organization

### Saving Modes

#### Case Mode (`saving_mode="case"`)
Creates a separate folder for each volume (recommended):
```
output_image_path/
├── patient_001/
│   ├── patient_001_registered.nii.gz
│   ├── patient_001_gaussian_filtered.nii.gz
│   └── patient_001_transformation.tfm
├── patient_002/
│   ├── patient_002_registered.nii.gz
│   ├── patient_002_gaussian_filtered.nii.gz
│   └── patient_002_transformation.tfm
```

#### Folder Mode (`saving_mode="folder"`)
Separates images and transformations:
```
output_image_path/
├── patient_001_registered.nii.gz
├── patient_001_gaussian_filtered.nii.gz
├── patient_002_registered.nii.gz
└── patient_002_gaussian_filtered.nii.gz

output_transformation_path/
├── patient_001_transformation.tfm
└── patient_002_transformation.tfm
```

### Output Files

For each input volume:

| File                              | Description                                       |
|-----------------------------------|---------------------------------------------------|
| `<PREFIX>_registered.nii.gz`      | Volume aligned to template space                  |
| `<PREFIX>_gaussian_filtered.nii.gz` | Preprocessed volume (removed if `cleanup=True`) |
| `<PREFIX>_transformation.tfm`     | Transformation parameters for reapplication       |

## Registration Pipeline

The registration process for each volume:

1. **Gaussian Filtering**
   - Smooths volume to reduce noise
   - Clips negative and extreme values
   - Prepares data for robust registration

2. **Mask-Based Registration**
   - Uses mutual information metric
   - Constrains optimization to brain regions
   - Moment-based initialization for alignment
   - Gradient descent optimization

3. **Transform Application**
   - Applies computed transformation
   - Resamples volume to template space
   - Preserves intensity characteristics

4. **Save Outputs**
   - Registered volume
   - Transformation file (for later reuse)
   - Optional intermediate files

## Registration Method

**Metric**: Mattes Mutual Information
- Robust to intensity differences between modalities
- Suitable for mono-modal (same imaging type) registration
- Handles intensity variations across scanners

**Optimization**: Gradient Descent
- Iteratively refines alignment
- Balances speed and accuracy
- Uses moment-based initialization

**Masking**: Brain region constraint
- Focuses registration on relevant anatomy
- Ignores background and skull
- Improves robustness and accuracy

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |
| `ValueError`         | Invalid `saving_mode` parameter                                    |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **Mask Requirement**: Each volume must have a corresponding mask with matching filename
- **Template Selection**: Choose a representative template from your dataset
- **Sequential Processing**: Volumes are registered one at a time
- **Progress Display**: Shows progress bar during processing
- **Output Directories**: Automatically created if they don't exist

## Examples

### Basic Usage - Case Mode
Register all volumes with per-case organization:

```python
from nidataset.preprocessing import register_CTA_dataset

register_CTA_dataset(
    nii_folder="dataset/scans/",
    mask_folder="dataset/brain_masks/",
    template_path="templates/standard_template.nii.gz",
    template_mask_path="templates/standard_mask.nii.gz",
    output_image_path="dataset/registered/",
    saving_mode="case"
)
# Creates: dataset/registered/case_001/case_001_registered.nii.gz, ...
```

### With Cleanup
Remove intermediate files to save space:

```python
register_CTA_dataset(
    nii_folder="data/raw_cta/",
    mask_folder="data/cta_masks/",
    template_path="template/atlas.nii.gz",
    template_mask_path="template/atlas_mask.nii.gz",
    output_image_path="data/aligned/",
    saving_mode="case",
    cleanup=True,  # Remove Gaussian-filtered intermediates
    debug=True
)
# Only registered volumes and transformations are kept
```

### Folder Mode Organization
Separate images and transformations:

```python
register_CTA_dataset(
    nii_folder="scans/",
    mask_folder="masks/",
    template_path="reference.nii.gz",
    template_mask_path="reference_mask.nii.gz",
    output_image_path="results/images/",
    output_transformation_path="results/transforms/",
    saving_mode="folder",
    debug=True
)
# Images in results/images/, transforms in results/transforms/
```

### Quality Control Workflow
Register and verify alignment quality:

```python
from nidataset.preprocessing import register_CTA_dataset
import nibabel as nib
import numpy as np

# Register dataset
register_CTA_dataset(
    nii_folder="qa/scans/",
    mask_folder="qa/masks/",
    template_path="qa/template.nii.gz",
    template_mask_path="qa/template_mask.nii.gz",
    output_image_path="qa/registered/",
    saving_mode="case",
    debug=True
)

# Load template and a registered volume
template = nib.load("qa/template.nii.gz")
template_data = template.get_fdata()

registered = nib.load("qa/registered/sample/sample_registered.nii.gz")
registered_data = registered.get_fdata()

# Verify shapes match
print(f"\nQuality Control:")
print(f"  Template shape: {template_data.shape}")
print(f"  Registered shape: {registered_data.shape}")
print(f"  Shapes match: {template_data.shape == registered_data.shape}")

# Check alignment in a specific region
roi = template_data[100:150, 100:150, 50:100]
roi_reg = registered_data[100:150, 100:150, 50:100]

# Calculate correlation as alignment metric
correlation = np.corrcoef(roi.flatten(), roi_reg.flatten())[0, 1]
print(f"  ROI correlation: {correlation:.3f}")
print(f"  Good alignment: correlation > 0.7")
```

### Creating Custom Template
Select a representative scan as template:

```python
import nibabel as nib
import shutil
from nidataset.preprocessing import register_CTA_dataset

# Step 1: Select template (e.g., scan with median brain size)
scan_files = ["scan_001.nii.gz", "scan_002.nii.gz", "scan_003.nii.gz"]
volumes = []

for scan in scan_files:
    img = nib.load(f"scans/{scan}")
    mask = nib.load(f"masks/{scan}")
    mask_data = mask.get_fdata()
    brain_volume = np.sum(mask_data > 0)
    volumes.append((scan, brain_volume))

# Sort by volume and pick median
volumes.sort(key=lambda x: x[1])
template_scan = volumes[len(volumes)//2][0]
print(f"Selected template: {template_scan}")

# Step 2: Copy to template folder
shutil.copy(f"scans/{template_scan}", "templates/custom_template.nii.gz")
shutil.copy(f"masks/{template_scan}", "templates/custom_template_mask.nii.gz")

# Step 3: Register all scans to this template
register_CTA_dataset(
    nii_folder="scans/",
    mask_folder="masks/",
    template_path="templates/custom_template.nii.gz",
    template_mask_path="templates/custom_template_mask.nii.gz",
    output_image_path="registered/",
    saving_mode="case",
    cleanup=True,
    debug=True
)
```

### Multi-Center Dataset Harmonization
Align scans from different sites:

```python
from nidataset.preprocessing import register_CTA_dataset

centers = ["center_A", "center_B", "center_C"]

# Use same template for all centers
template = "standard_atlas/template.nii.gz"
template_mask = "standard_atlas/template_mask.nii.gz"

for center in centers:
    print(f"\nProcessing {center}...")
    
    register_CTA_dataset(
        nii_folder=f"data/{center}/scans/",
        mask_folder=f"data/{center}/masks/",
        template_path=template,
        template_mask_path=template_mask,
        output_image_path=f"harmonized/{center}/",
        saving_mode="case",
        cleanup=True,
        debug=True
    )

print("\nAll centers registered to common template")
```

### Parallel Processing Strategy
Process large datasets efficiently:

```python
from nidataset.preprocessing import register_CTA_dataset
import multiprocessing as mp
import os

def register_subset(file_list, subset_id):
    """Register a subset of files."""
    # Create temporary folder with subset
    temp_folder = f"temp_subset_{subset_id}/"
    os.makedirs(temp_folder, exist_ok=True)
    
    for f in file_list:
        shutil.copy(f"all_scans/{f}", temp_folder)
    
    # Register subset
    register_CTA_dataset(
        nii_folder=temp_folder,
        mask_folder="all_masks/",
        template_path="template.nii.gz",
        template_mask_path="template_mask.nii.gz",
        output_image_path=f"registered_subset_{subset_id}/",
        saving_mode="case",
        cleanup=True
    )
    
    # Cleanup
    shutil.rmtree(temp_folder)

# Split files into subsets
all_files = [f for f in os.listdir("all_scans/") if f.endswith('.nii.gz')]
n_cores = 4
subsets = np.array_split(all_files, n_cores)

# Process in parallel
with mp.Pool(n_cores) as pool:
    pool.starmap(register_subset, enumerate(subsets))

# Merge results
final_output = "registered_all/"
os.makedirs(final_output, exist_ok=True)
for i in range(n_cores):
    shutil.copytree(f"registered_subset_{i}/", final_output, dirs_exist_ok=True)

print("Parallel registration complete")
```

### Applying Saved Transformations
Reuse transformations for other data:

```python
import SimpleITK as sitk
from nidataset.preprocessing import register_CTA_dataset

# Step 1: Register structural scans
register_CTA_dataset(
    nii_folder="structural/",
    mask_folder="masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_image_path="registered_structural/",
    saving_mode="case",
    cleanup=False,
    debug=True
)

# Step 2: Apply same transformations to functional scans
transform_folder = "registered_structural/"
functional_folder = "functional/"

for case in os.listdir(transform_folder):
    if os.path.isdir(os.path.join(transform_folder, case)):
        # Load transformation
        tfm_file = f"{transform_folder}/{case}/{case}_transformation.tfm"
        transform = sitk.ReadTransform(tfm_file)
        
        # Load functional scan
        func_file = f"{functional_folder}/{case}.nii.gz"
        func_img = sitk.ReadImage(func_file)
        
        # Apply transformation
        template_img = sitk.ReadImage("template.nii.gz")
        registered_func = sitk.Resample(
            func_img,
            template_img,
            transform,
            sitk.sitkLinear,
            0.0
        )
        
        # Save
        output_file = f"registered_functional/{case}_registered.nii.gz"
        os.makedirs("registered_functional/", exist_ok=True)
        sitk.WriteImage(registered_func, output_file)

print("Transformations applied to functional scans")
```

### Assessing Registration Quality
Evaluate alignment across dataset:

```python
from nidataset.preprocessing import register_CTA_dataset
import nibabel as nib
import numpy as np
import pandas as pd

# Register dataset
register_CTA_dataset(
    nii_folder="scans/",
    mask_folder="masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_image_path="registered/",
    saving_mode="case",
    debug=True
)

# Load template
template = nib.load("template.nii.gz")
template_data = template.get_fdata()
template_mask = nib.load("template_mask.nii.gz").get_fdata()

# Assess each registered volume
results = []
for case_folder in os.listdir("registered/"):
    case_path = os.path.join("registered/", case_folder)
    if os.path.isdir(case_path):
        reg_file = f"{case_path}/{case_folder}_registered.nii.gz"
        reg_img = nib.load(reg_file)
        reg_data = reg_img.get_fdata()
        
        # Calculate metrics within brain mask
        mask_indices = template_mask > 0
        template_roi = template_data[mask_indices]
        registered_roi = reg_data[mask_indices]
        
        # Correlation
        correlation = np.corrcoef(template_roi, registered_roi)[0, 1]
        
        # Normalized mutual information (approximate)
        hist, _, _ = np.histogram2d(
            template_roi.flatten(),
            registered_roi.flatten(),
            bins=50
        )
        pxy = hist / hist.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        
        # MI calculation
        px_py = px[:, None] * py[None, :]
        nz = pxy > 0
        mi = np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))
        
        results.append({
            'case': case_folder,
            'correlation': correlation,
            'mutual_information': mi
        })

# Summary
df = pd.DataFrame(results)
print("\nRegistration Quality Assessment:")
print(f"  Mean correlation: {df['correlation'].mean():.3f}")
print(f"  Min correlation: {df['correlation'].min():.3f}")
print(f"  Cases with correlation < 0.7: {(df['correlation'] < 0.7).sum()}")

# Identify poor registrations
poor_reg = df[df['correlation'] < 0.7]
if not poor_reg.empty:
    print(f"\nCases needing review:")
    print(poor_reg[['case', 'correlation']])
```

### Creating Dataset Statistics
Track registration parameters:

```python
from nidataset.preprocessing import register_CTA_dataset
import SimpleITK as sitk
import pandas as pd

# Register with debug to see details
register_CTA_dataset(
    nii_folder="data/scans/",
    mask_folder="data/masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_image_path="data/registered/",
    saving_mode="case",
    cleanup=False,
    debug=True
)

# Extract transformation parameters
transform_stats = []
for case in os.listdir("data/registered/"):
    if os.path.isdir(f"data/registered/{case}"):
        tfm_file = f"data/registered/{case}/{case}_transformation.tfm"
        
        # Read transform
        transform = sitk.ReadTransform(tfm_file)
        params = transform.GetParameters()
        
        # Extract translation components (assuming affine)
        if len(params) >= 12:
            tx, ty, tz = params[9:12]
            transform_stats.append({
                'case': case,
                'translation_x': tx,
                'translation_y': ty,
                'translation_z': tz,
                'total_translation': np.sqrt(tx**2 + ty**2 + tz**2)
            })

# Analyze
df = pd.DataFrame(transform_stats)
print("\nRegistration Statistics:")
print(f"  Mean translation: {df['total_translation'].mean():.2f} mm")
print(f"  Max translation: {df['total_translation'].max():.2f} mm")
print(f"  Cases with large shifts (>20mm): {(df['total_translation'] > 20).sum()}")
```

## Typical Workflow

```python
from nidataset.preprocessing import register_CTA_dataset
import nibabel as nib

# 1. Prepare inputs
scan_folder = "dataset/angiography_scans/"
mask_folder = "dataset/brain_masks/"
template = "atlas/standard_template.nii.gz"
template_mask = "atlas/standard_mask.nii.gz"

# 2. Register all scans to template
register_CTA_dataset(
    nii_folder=scan_folder,
    mask_folder=mask_folder,
    template_path=template,
    template_mask_path=template_mask,
    output_image_path="dataset/registered/",
    saving_mode="case",
    cleanup=True,  # Save space
    debug=True
)

# 3. Verify a sample result
template_img = nib.load(template)
sample_reg = nib.load("dataset/registered/sample/sample_registered.nii.gz")

print(f"Template shape: {template_img.shape}")
print(f"Registered shape: {sample_reg.shape}")

# 4. Use registered volumes for:
# - Voxel-wise analysis across subjects
# - Creating population atlases
# - Group comparisons
# - Machine learning with spatial features
```