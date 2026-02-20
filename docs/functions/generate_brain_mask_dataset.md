---
title: generate_brain_mask_dataset
parent: Package Functions
nav_order: 21
---

# `generate_brain_mask_dataset`

Batch generate brain tissue masks for all medical imaging volumes in a dataset folder using intensity-based segmentation and morphological refinement.

```python
generate_brain_mask_dataset(
    nii_folder: str,
    output_path: str,
    threshold: tuple = None,
    closing_radius: int = 3,
    debug: bool = False
) -> None
```

## Overview

This function processes all volumes in a dataset folder to generate binary masks that isolate brain tissue from background and skull. It applies intensity thresholding followed by morphological closing to create clean, connected tissue masks.

The segmentation process:
1. **Thresholding**: Applies intensity-based segmentation (manual or automatic Otsu)
2. **Morphological closing**: Fills small gaps and smooths boundaries
3. **Binary mask creation**: Generates clean tissue/background separation

This is particularly useful for:
- Preprocessing medical imaging datasets
- Removing non-brain tissue before analysis
- Standardizing skull stripping across datasets
- Quality control of scan coverage

## Parameters

| Name             | Type              | Default | Description                                                                                          |
|------------------|-------------------|---------|------------------------------------------------------------------------------------------------------|
| `nii_folder`     | `str`             | *required* | Path to the directory containing medical volumes in `.nii.gz` format.                             |
| `output_path`    | `str`             | *required* | Directory where brain masks will be saved. Created automatically if it doesn't exist.             |
| `threshold`      | `tuple` or `None` | `None`  | Intensity range `(low, high)` for segmentation. If `None`, uses automatic Otsu thresholding.        |
| `closing_radius` | `int`             | `3`     | Radius for morphological closing operation to refine mask boundaries.                                |
| `debug`          | `bool`            | `False` | If `True`, prints processing summary with total number of masks generated.                           |

## Returns

`None` – The function saves binary mask files to disk.

## Output Files

Each input volume generates an output mask:
```
<PREFIX>_mask.nii.gz
```
where `<PREFIX>` is the original filename without the `.nii.gz` extension.

**Example**: Input `scan_001.nii.gz` → Output `scan_001_mask.nii.gz`

### Output Mask Properties
- **Data type**: Binary mask (0 for background, 1 for brain tissue)
- **Dimensions**: Same as input volume
- **Spatial metadata**: Inherits affine transformation from input

## Thresholding Modes

### Automatic Thresholding (`threshold=None`)
Uses Otsu's method to automatically determine optimal threshold for each volume:
- Adapts to different intensity ranges across scans
- No manual tuning required
- Suitable for datasets with varying contrast

### Manual Thresholding (`threshold=(low, high)`)
Uses fixed intensity bounds for all volumes:
- Consistent segmentation across dataset
- Better control over tissue selection
- Requires knowledge of intensity ranges

**Example intensity ranges**:
- CT scans: `(20, 100)` for soft tissue, `(50, 300)` for brain with contrast
- MRI T1: `(50, 200)` typical brain intensities
- MRI T2: `(100, 500)` typical brain intensities

## Morphological Closing

The `closing_radius` parameter controls the morphological closing operation:
- **Smaller radius (1-2)**: Minimal smoothing, preserves fine details
- **Medium radius (3-4)**: Balanced smoothing, fills small gaps
- **Larger radius (5+)**: Aggressive smoothing, may lose small structures

**Effect**: Closing fills holes and connects nearby regions, creating smoother, more continuous masks.

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **3D Volumes Required**: Input must be 3D NIfTI images
- **Progress Display**: Shows progress bar during processing
- **Error Handling**: Individual file failures are reported but don't stop batch processing
- **Output Directory**: Automatically created if it doesn't exist
- **Threshold Selection**: Start with automatic mode, then use manual if consistency is needed

## Examples

### Basic Usage - Automatic Thresholding
Generate masks with automatic Otsu thresholding:

```python
from nidataset.volume import generate_brain_mask_dataset

generate_brain_mask_dataset(
    nii_folder="dataset/scans/",
    output_path="dataset/brain_masks/",
    threshold=None,
    closing_radius=3
)
# Creates: dataset/brain_masks/<scan>_mask.nii.gz for each scan
```

### Manual Thresholding
Use fixed intensity bounds for consistent segmentation:

```python
generate_brain_mask_dataset(
    nii_folder="ct_scans/",
    output_path="ct_masks/",
    threshold=(50, 300),  # Intensity range for brain tissue
    closing_radius=3,
    debug=True
)
# Prints summary: Total brain masks generated: 150
```

### Fine-Tuned Morphological Closing
Adjust closing radius for different tissue characteristics:

```python
# Conservative closing for detailed structures
generate_brain_mask_dataset(
    nii_folder="high_res_scans/",
    output_path="detailed_masks/",
    threshold=(40, 200),
    closing_radius=2,  # Minimal smoothing
    debug=True
)

# Aggressive closing for noisy data
generate_brain_mask_dataset(
    nii_folder="noisy_scans/",
    output_path="smooth_masks/",
    threshold=(50, 250),
    closing_radius=5,  # Strong smoothing
    debug=True
)
```

### Processing Multiple Datasets
Generate masks for different scan types with optimized parameters:

```python
from nidataset.volume import generate_brain_mask_dataset

datasets = {
    'ct_contrast': {
        'folder': 'data/ct_with_contrast/',
        'threshold': (50, 300),
        'radius': 3
    },
    'ct_plain': {
        'folder': 'data/ct_plain/',
        'threshold': (20, 100),
        'radius': 4
    },
    'mri_t1': {
        'folder': 'data/mri_t1/',
        'threshold': None,  # Automatic
        'radius': 3
    }
}

for name, config in datasets.items():
    print(f"Processing {name}...")
    generate_brain_mask_dataset(
        nii_folder=config['folder'],
        output_path=f"masks/{name}/",
        threshold=config['threshold'],
        closing_radius=config['radius'],
        debug=True
    )
```

### Quality Control Workflow
Generate masks and verify quality:

```python
import nibabel as nib
import numpy as np
from nidataset.volume import generate_brain_mask_dataset

# Generate masks
generate_brain_mask_dataset(
    nii_folder="qa/scans/",
    output_path="qa/masks/",
    threshold=(50, 300),
    closing_radius=3,
    debug=True
)

# Verify a sample mask
sample_scan = nib.load("qa/scans/sample.nii.gz")
sample_mask = nib.load("qa/masks/sample_mask.nii.gz")

scan_data = sample_scan.get_fdata()
mask_data = sample_mask.get_fdata()

# Calculate coverage
total_voxels = np.prod(scan_data.shape)
brain_voxels = np.sum(mask_data > 0)
coverage = (brain_voxels / total_voxels) * 100

print(f"\nQuality Control:")
print(f"  Original volume: {scan_data.shape}")
print(f"  Mask volume: {mask_data.shape}")
print(f"  Brain coverage: {coverage:.1f}%")
print(f"  Mask values: {np.unique(mask_data)}")

# Typical brain coverage: 30-50% for head scans
if coverage < 20 or coverage > 60:
    print("  Warning: Unusual coverage, check threshold settings")
```

### Determining Optimal Threshold
Test different thresholds to find the best settings:

```python
import nibabel as nib
from nidataset.volume import generate_brain_mask_dataset

# Test different threshold ranges
test_thresholds = [
    (30, 200),
    (50, 250),
    (70, 300),
    None  # Automatic
]

sample_scan = "test_data/sample.nii.gz"

for i, thresh in enumerate(test_thresholds):
    output_folder = f"threshold_test/test_{i}/"
    
    generate_brain_mask_dataset(
        nii_folder="test_data/",
        output_path=output_folder,
        threshold=thresh,
        closing_radius=3,
        debug=True
    )
    
    # Load and analyze result
    mask = nib.load(f"{output_folder}/sample_mask.nii.gz")
    mask_data = mask.get_fdata()
    coverage = np.sum(mask_data > 0) / np.prod(mask_data.shape) * 100
    
    thresh_str = f"{thresh[0]}-{thresh[1]}" if thresh else "Otsu"
    print(f"Threshold {thresh_str}: Coverage = {coverage:.1f}%")

print("\nVisually inspect outputs to choose best threshold")
```

### Integration with Preprocessing Pipeline
Use masks to crop volumes to brain region:

```python
import nibabel as nib
import numpy as np
from nidataset.volume import generate_brain_mask_dataset

# Generate masks
generate_brain_mask_dataset(
    nii_folder="raw_scans/",
    output_path="brain_masks/",
    threshold=(50, 300),
    closing_radius=3,
    debug=True
)

# Apply masks to crop original scans
scan_folder = "raw_scans/"
mask_folder = "brain_masks/"
output_folder = "masked_scans/"
os.makedirs(output_folder, exist_ok=True)

for scan_file in os.listdir(scan_folder):
    if scan_file.endswith('.nii.gz'):
        # Load scan and mask
        scan = nib.load(os.path.join(scan_folder, scan_file))
        scan_data = scan.get_fdata()
        
        mask_file = scan_file.replace('.nii.gz', '_mask.nii.gz')
        mask = nib.load(os.path.join(mask_folder, mask_file))
        mask_data = mask.get_fdata()
        
        # Apply mask
        masked_data = scan_data * mask_data
        
        # Save masked scan
        masked_img = nib.Nifti1Image(masked_data, scan.affine)
        output_path = os.path.join(output_folder, scan_file)
        nib.save(masked_img, output_path)

print(f"Created {len(os.listdir(output_folder))} masked scans")
```

### Comparing Automatic vs Manual Thresholding
Evaluate both methods on your dataset:

```python
import nibabel as nib
import numpy as np
from nidataset.volume import generate_brain_mask_dataset

# Generate with automatic thresholding
generate_brain_mask_dataset(
    nii_folder="comparison/scans/",
    output_path="comparison/auto_masks/",
    threshold=None,  # Automatic
    closing_radius=3,
    debug=True
)

# Generate with manual thresholding
generate_brain_mask_dataset(
    nii_folder="comparison/scans/",
    output_path="comparison/manual_masks/",
    threshold=(50, 300),  # Manual
    closing_radius=3,
    debug=True
)

# Compare results
scan_files = [f for f in os.listdir("comparison/scans/") if f.endswith('.nii.gz')]

for scan_file in scan_files:
    prefix = scan_file.replace('.nii.gz', '')
    
    auto_mask = nib.load(f"comparison/auto_masks/{prefix}_mask.nii.gz")
    manual_mask = nib.load(f"comparison/manual_masks/{prefix}_mask.nii.gz")
    
    auto_data = auto_mask.get_fdata()
    manual_data = manual_mask.get_fdata()
    
    auto_coverage = np.sum(auto_data > 0) / np.prod(auto_data.shape) * 100
    manual_coverage = np.sum(manual_data > 0) / np.prod(manual_data.shape) * 100
    
    # Calculate overlap (Dice coefficient)
    intersection = np.sum((auto_data > 0) & (manual_data > 0))
    union = np.sum(auto_data > 0) + np.sum(manual_data > 0)
    dice = 2 * intersection / union if union > 0 else 0
    
    print(f"\n{prefix}:")
    print(f"  Auto coverage: {auto_coverage:.1f}%")
    print(f"  Manual coverage: {manual_coverage:.1f}%")
    print(f"  Dice overlap: {dice:.3f}")
```

### Batch Processing with Error Handling
Process large datasets with robust error handling:

```python
from nidataset.volume import generate_brain_mask_dataset
import logging

# Setup logging
logging.basicConfig(
    filename='mask_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

try:
    logging.info("Starting brain mask generation")
    
    generate_brain_mask_dataset(
        nii_folder="large_dataset/scans/",
        output_path="large_dataset/masks/",
        threshold=(50, 300),
        closing_radius=3,
        debug=True
    )
    
    logging.info("Brain mask generation completed successfully")
    
except FileNotFoundError as e:
    logging.error(f"File error: {e}")
    print(f"Error: {e}")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    print(f"Error: {e}")
finally:
    logging.info("Processing finished")
```

### Creating Visualization Overlays
Generate masks and create overlay images:

```python
import nibabel as nib
import numpy as np
from nidataset.volume import generate_brain_mask_dataset

# Generate masks
generate_brain_mask_dataset(
    nii_folder="visualization/scans/",
    output_path="visualization/masks/",
    threshold=(50, 300),
    closing_radius=3
)

# Create overlay for visualization
scan = nib.load("visualization/scans/example.nii.gz")
mask = nib.load("visualization/masks/example_mask.nii.gz")

scan_data = scan.get_fdata()
mask_data = mask.get_fdata()

# Create overlay (mask boundary highlighted)
overlay = scan_data.copy()
# Highlight mask edges
from scipy import ndimage
edges = ndimage.sobel(mask_data)
overlay[edges > 0] = scan_data.max()

# Save overlay
overlay_img = nib.Nifti1Image(overlay, scan.affine)
nib.save(overlay_img, "visualization/overlay.nii.gz")
print("Overlay created for visual inspection")
```

## Typical Workflow

```python
from nidataset.volume import generate_brain_mask_dataset
import nibabel as nib
import numpy as np

# 1. Define input and output paths
scan_folder = "dataset/raw_scans/"
mask_output = "dataset/brain_masks/"

# 2. Generate masks with appropriate settings
generate_brain_mask_dataset(
    nii_folder=scan_folder,
    output_path=mask_output,
    threshold=(50, 300),  # Adjust based on your imaging modality
    closing_radius=3,
    debug=True
)

# 3. Verify a sample result
sample_scan = nib.load("dataset/raw_scans/sample.nii.gz")
sample_mask = nib.load("dataset/brain_masks/sample_mask.nii.gz")

scan_data = sample_scan.get_fdata()
mask_data = sample_mask.get_fdata()

coverage = np.sum(mask_data > 0) / np.prod(mask_data.shape) * 100
print(f"\nMask quality check:")
print(f"  Brain coverage: {coverage:.1f}%")
print(f"  Expected range: 30-50% for head scans")

# 4. Use masks for:
# - Skull stripping
# - Region of interest extraction
# - Preprocessing before analysis
# - Quality control
```