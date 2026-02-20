---
title: register_CTA_dataset
parent: Package Functions
nav_order: 32
---

# `register_CTA_dataset`

Batch register all medical imaging volumes in a dataset folder to a reference template using intensity-based registration with mutual information.

```python
register_CTA_dataset(
    nii_folder: str,
    mask_folder: str,
    template_path: str,
    template_mask_path: str,
    output_path: str,
    saving_mode: str = "case",
    cleanup: bool = False,
    debug: bool = False,
    number_histogram_bins: int = 128,
    learning_rate: float = 0.0001,
    number_iterations: int = 2000,
    initialization_strategy: int = sitk.CenteredTransformInitializerFilter.MOMENTS,
    sigma_first: float = 2.0,
    sigma_second: float = 3.0,
    metric_sampling_percentage: float = 0.5,
    initial_transform = None
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
| `output_path`                | `str`  | *required* | Base directory for all outputs. Structure depends on `saving_mode`.                                 |
| `saving_mode`                | `str`  | `"case"`   | Organization mode: `"case"` (folder per volume) or `"folder"` (shared folders).                    |
| `cleanup`                    | `bool` | `False`    | If `True`, deletes intermediate Gaussian-filtered files after registration.                         |
| `debug`                      | `bool` | `False`    | If `True`, prints detailed registration information for each volume.                                |
| `number_histogram_bins`      | `int`  | `128`      | Number of histogram bins for Mattes Mutual Information metric. Common values: 10, 50, 64, 128.     |
| `learning_rate`              | `float`| `0.0001`   | Learning rate for Gradient Descent optimizer. Common values: 0.0001-1.0.                            |
| `number_iterations`          | `int`  | `2000`     | Maximum number of optimization iterations. Common values: 500-5000.                                 |
| `initialization_strategy`    | `int`  | `MOMENTS`  | Strategy for initializing transformation: `MOMENTS` (center of mass) or `GEOMETRY` (center/orientation). |
| `sigma_first`                | `float`| `2.0`      | Standard deviation for the first Gaussian smoothing filter.                                         |
| `sigma_second`               | `float`| `3.0`      | Standard deviation for the second Gaussian smoothing filter.                                        |
| `metric_sampling_percentage` | `float`| `0.5`      | Percentage of voxels to sample for metric evaluation (0.0-1.0). Default: 0.5 (50%).               |
| `initial_transform`          | `None` | `None`     | Initial transformation object. If `None`, defaults to `sitk.Euler3DTransform()`.                   |

## Returns

`None` – The function saves registered volumes and transformations to disk.

## Output Organization

### Saving Modes

The function supports two organizational strategies for output files:

#### Case Mode (`saving_mode="case"`)
Creates a separate folder for each volume (recommended for dataset organization):

**Without cleanup** (`cleanup=False`):
```
output_path/
├── patient_001/
│   ├── patient_001_registered.nii.gz
│   ├── patient_001_gaussian_filtered.nii.gz  ← intermediate file kept
│   └── patient_001_transformation.tfm
├── patient_002/
│   ├── patient_002_registered.nii.gz
│   ├── patient_002_gaussian_filtered.nii.gz  ← intermediate file kept
│   └── patient_002_transformation.tfm
```

**With cleanup** (`cleanup=True`):
```
output_path/
├── patient_001/
│   ├── patient_001_registered.nii.gz
│   └── patient_001_transformation.tfm  ← gaussian_filtered.nii.gz deleted
├── patient_002/
│   ├── patient_002_registered.nii.gz
│   └── patient_002_transformation.tfm  ← gaussian_filtered.nii.gz deleted
```

#### Folder Mode (`saving_mode="folder"`)
Separates images and transformations into dedicated subdirectories. This mode uses temporary directories during processing.

**Processing flow**:
1. For each volume, creates a temporary directory: `output_path/_temp_<PREFIX>/`
2. Runs registration, generating files in the temporary directory
3. Moves `<PREFIX>_registered.nii.gz` to `output_path/registered/`
4. Moves `<PREFIX>_transformation.tfm` to `output_path/transforms/`
5. If `cleanup=True`, deletes the temporary directory
6. If `cleanup=False`, keeps the temporary directory with gaussian-filtered files

**Without cleanup** (`cleanup=False`):
```
output_path/
├── _temp_patient_001/
│   └── patient_001_gaussian_filtered.nii.gz  ← intermediate kept in temp
├── _temp_patient_002/
│   └── patient_002_gaussian_filtered.nii.gz  ← intermediate kept in temp
├── _temp_patient_003/
│   └── patient_003_gaussian_filtered.nii.gz  ← intermediate kept in temp
├── registered/
│   ├── patient_001_registered.nii.gz
│   ├── patient_002_registered.nii.gz
│   └── patient_003_registered.nii.gz
└── transforms/
    ├── patient_001_transformation.tfm
    ├── patient_002_transformation.tfm
    └── patient_003_transformation.tfm
```

**With cleanup** (`cleanup=True`):
```
output_path/
├── registered/
│   ├── patient_001_registered.nii.gz
│   ├── patient_002_registered.nii.gz
│   └── patient_003_registered.nii.gz
└── transforms/
    ├── patient_001_transformation.tfm
    ├── patient_002_transformation.tfm
    └── patient_003_transformation.tfm
```

**Important notes for folder mode**:
- Temporary directories (`_temp_<PREFIX>/`) are created for each volume during processing
- If `cleanup=True`, temporary directories are deleted after moving registered images and transformations
- If `cleanup=False`, temporary directories are kept with their gaussian-filtered intermediate files
- This allows you to inspect preprocessing steps or recover gaussian-filtered volumes if needed

## Output Files

For each input volume, the function generates:

| File                              | Description                                       | Kept After Cleanup |
|-----------------------------------|---------------------------------------------------|--------------------|
| `<PREFIX>_registered.nii.gz`      | Volume aligned to template space                  | Yes                |
| `<PREFIX>_gaussian_filtered.nii.gz` | Preprocessed volume used for registration       | No (if cleanup=True) |
| `<PREFIX>_transformation.tfm`     | Transformation parameters                         | Yes                |

**Example**: Input `scan_042.nii.gz` produces:
- `scan_042_registered.nii.gz`
- `scan_042_gaussian_filtered.nii.gz` (temporary)
- `scan_042_transformation.tfm`

## Registration Pipeline

The registration process for each volume:

1. **Gaussian Filtering**
   - Smooths volume to reduce noise
   - Clips negative and extreme values
   - Prepares data for robust registration

2. **Mask-Based Registration**
   - Uses mutual information metric
   - Constrains optimization to brain regions
   - Moment-based or geometry-based initialization
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
- **Histogram bins**: Configurable (default: 128)
- **Sampling**: Configurable percentage (default: 50%)
- Robust to intensity differences between modalities
- Suitable for mono-modal (same imaging type) registration
- Handles intensity variations across scanners

**Optimization**: Gradient Descent
- **Learning rate**: Configurable (default: 0.0001)
- **Iterations**: Configurable maximum (default: 2000)
- Iteratively refines alignment
- Balances speed and accuracy
- Uses moment-based or geometry-based initialization

**Transform**: Euler3D (default) or custom
- 6 degrees of freedom (rigid transformation)
- Can be customized via `initial_transform` parameter

**Masking**: Brain region constraint
- Focuses registration on relevant anatomy
- Ignores background and skull
- Improves robustness and accuracy

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |
| `ValueError`         | Invalid `saving_mode` parameter (must be `"case"` or `"folder"`)  |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **Mask Requirement**: Each volume must have a corresponding mask with matching filename in `mask_folder`
- **Template Selection**: Choose a representative template from your dataset
- **Sequential Processing**: Volumes are registered one at a time with a progress bar (tqdm)
- **Output Directories**: Automatically created if they don't exist
- **Progress Display**: Shows real-time progress during batch processing
- **Parameter Tuning**: All registration parameters can be customized for dataset-specific optimization

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
    output_path="dataset/registered/",
    saving_mode="case"
)
# Creates: dataset/registered/case_001/case_001_registered.nii.gz, ...
```

### Custom Registration Parameters
Fine-tune registration for your dataset:

```python
import SimpleITK as sitk
from nidataset.preprocessing import register_CTA_dataset

register_CTA_dataset(
    nii_folder="dataset/scans/",
    mask_folder="dataset/masks/",
    template_path="template/CTA_template.nii.gz",
    template_mask_path="template/CTA_template_mask.nii.gz",
    output_path="dataset/registered/",
    saving_mode="case",
    number_histogram_bins=64,
    learning_rate=0.01,
    number_iterations=1000,
    initialization_strategy=sitk.CenteredTransformInitializerFilter.GEOMETRY,
    sigma_first=1.5,
    sigma_second=2.5,
    metric_sampling_percentage=0.7,
    cleanup=True,
    debug=True
)
```

### With Cleanup
Remove intermediate files to save space:

```python
register_CTA_dataset(
    nii_folder="data/raw_cta/",
    mask_folder="data/cta_masks/",
    template_path="template/atlas.nii.gz",
    template_mask_path="template/atlas_mask.nii.gz",
    output_path="data/aligned/",
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
    output_path="results/",
    saving_mode="folder",
    debug=True
)
# Images in results/registered/, transforms in results/transforms/
```

### Using Custom Initial Transform
Start with an affine transform instead of rigid:

```python
import SimpleITK as sitk
from nidataset.preprocessing import register_CTA_dataset

# Create a custom initial transform
affine_transform = sitk.AffineTransform(3)

register_CTA_dataset(
    nii_folder="scans/",
    mask_folder="masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_path="registered/",
    saving_mode="case",
    initial_transform=affine_transform,
    number_iterations=3000,  # More iterations for affine
    debug=True
)
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
    output_path="qa/registered/",
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
import numpy as np
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
    output_path="registered/",
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
        output_path=f"harmonized/{center}/",
        saving_mode="case",
        cleanup=True,
        debug=True
    )

print("\nAll centers registered to common template")
```

### Applying Saved Transformations
Reuse transformations for other data:

```python
import SimpleITK as sitk
import os
from nidataset.preprocessing import register_CTA_dataset

# Step 1: Register structural scans
register_CTA_dataset(
    nii_folder="structural/",
    mask_folder="masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_path="registered_structural/",
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
import os

# Register dataset
register_CTA_dataset(
    nii_folder="scans/",
    mask_folder="masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_path="registered/",
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
import numpy as np
import os

# Register with debug to see details
register_CTA_dataset(
    nii_folder="data/scans/",
    mask_folder="data/masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_path="data/registered/",
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
        
        # Extract translation components (for Euler3D: 6 params)
        if len(params) >= 6:
            rotations = params[0:3]
            translations = params[3:6]
            transform_stats.append({
                'case': case,
                'rotation_x': rotations[0],
                'rotation_y': rotations[1],
                'rotation_z': rotations[2],
                'translation_x': translations[0],
                'translation_y': translations[1],
                'translation_z': translations[2],
                'total_translation': np.sqrt(sum(t**2 for t in translations))
            })

# Analyze
df = pd.DataFrame(transform_stats)
print("\nRegistration Statistics:")
print(f"  Mean translation: {df['total_translation'].mean():.2f} mm")
print(f"  Max translation: {df['total_translation'].max():.2f} mm")
print(f"  Cases with large shifts (>20mm): {(df['total_translation'] > 20).sum()}")
```

### Comparing Different Registration Settings
Compare standard vs. high-quality registration:

```python
from nidataset.preprocessing import register_CTA_dataset
import nibabel as nib
import numpy as np
import os

# Standard registration
register_CTA_dataset(
    nii_folder="comparison/scans/",
    mask_folder="comparison/masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_path="comparison/standard/",
    saving_mode="case",
    cleanup=True
)

# High-quality registration
register_CTA_dataset(
    nii_folder="comparison/scans/",
    mask_folder="comparison/masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_path="comparison/high_quality/",
    saving_mode="case",
    number_histogram_bins=256,
    learning_rate=0.001,
    number_iterations=5000,
    metric_sampling_percentage=0.8,
    cleanup=True
)

# Compare results
template_data = nib.load("template.nii.gz").get_fdata()
template_mask = nib.load("template_mask.nii.gz").get_fdata()
mask_indices = template_mask > 0
template_roi = template_data[mask_indices]

results = []
for case in os.listdir("comparison/standard/"):
    if os.path.isdir(f"comparison/standard/{case}"):
        # Load both versions
        standard = nib.load(f"comparison/standard/{case}/{case}_registered.nii.gz")
        hq = nib.load(f"comparison/high_quality/{case}/{case}_registered.nii.gz")
        
        standard_roi = standard.get_fdata()[mask_indices]
        hq_roi = hq.get_fdata()[mask_indices]
        
        # Compare correlations
        corr_standard = np.corrcoef(template_roi, standard_roi)[0, 1]
        corr_hq = np.corrcoef(template_roi, hq_roi)[0, 1]
        
        results.append({
            'case': case,
            'standard_corr': corr_standard,
            'hq_corr': corr_hq,
            'improvement': corr_hq - corr_standard
        })

df = pd.DataFrame(results)
print(f"\nAverage improvement: {df['improvement'].mean():.3f}")
print(f"Cases improved: {(df['improvement'] > 0).sum()}/{len(df)}")
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
    output_path="dataset/registered/",
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

## Parameter Tuning Guide

| Parameter | Effect | Recommendations |
|-----------|--------|-----------------|
| `number_histogram_bins` | Higher values = finer intensity discretization | 64-128 for most cases; 256 for high-contrast images |
| `learning_rate` | Higher values = faster but less stable convergence | 0.0001-0.001 for standard; 0.01+ for fast initial alignment |
| `number_iterations` | More iterations = potential for better alignment | 1000-2000 standard; 3000-5000 for difficult cases |
| `metric_sampling_percentage` | Higher sampling = more accurate but slower | 0.3-0.5 for speed; 0.7-1.0 for accuracy |
| `sigma_first` / `sigma_second` | Controls smoothing strength | Lower for sharp features; higher for noisy images |
| `initialization_strategy` | MOMENTS vs GEOMETRY | MOMENTS for asymmetric anatomy; GEOMETRY for symmetric |
| `saving_mode` | Organization of outputs | "case" for per-subject analysis; "folder" for simpler structure |