---
title: swap_nifti_views
parent: Package Functions
nav_order: 29
---

# `swap_nifti_views`

Reorient a 3D medical imaging volume from one anatomical view to another by permuting axes, applying rotation, and updating spatial metadata.

```python
swap_nifti_views(
    nii_path: str,
    output_path: str,
    source_view: str,
    target_view: str,
    debug: bool = False
) -> None
```

## Overview

This function transforms the anatomical orientation of a 3D volume by reordering its axes and applying appropriate rotation. The transformation ensures that slicing along different axes produces the desired anatomical plane while maintaining correct spatial orientation through affine matrix updates.

**Transformation process**:
1. **Axis permutation**: Reorder dimensions to match target view
2. **90-degree rotation**: Align orientation correctly
3. **Affine update**: Preserve spatial coordinates and orientation
4. **Header update**: Adjust metadata for new dimensions

This is useful for:
- Standardizing anatomical orientations across datasets
- Converting between different acquisition orientations
- Preparing data for view-specific analysis
- Matching orientations for registration or comparison
- Creating consistent visualization views

## Parameters

| Name          | Type   | Default    | Description                                                                                          |
|---------------|--------|------------|------------------------------------------------------------------------------------------------------|
| `nii_path`    | `str`  | *required* | Path to the input volume in `.nii.gz` format.                                                       |
| `output_path` | `str`  | *required* | Directory where the reoriented volume will be saved. Created automatically if it doesn't exist.     |
| `source_view` | `str`  | *required* | Current anatomical orientation: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `target_view` | `str`  | *required* | Desired anatomical orientation: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `debug`       | `bool` | `False`    | If `True`, prints detailed information about shapes and transformations.                            |

## Returns

`None` – The function saves the reoriented volume to disk.

## Output File

The reoriented volume is saved as:
```
<PREFIX>_swapped_<SOURCE>_to_<TARGET>.nii.gz
```

**Examples**:
- Input `scan.nii.gz`, axial→sagittal → `scan_swapped_axial_to_sagittal.nii.gz`
- Input `brain.nii.gz`, coronal→axial → `brain_swapped_coronal_to_axial.nii.gz`

## Anatomical Views

### View Definitions

| View         | Primary Axis | Slice Direction | Typical Display          |
|--------------|--------------|-----------------|--------------------------|
| `"axial"`    | Z-axis       | Top to bottom   | Horizontal brain slices  |
| `"coronal"`  | Y-axis       | Front to back   | Frontal view             |
| `"sagittal"` | X-axis       | Left to right   | Side view                |

### Valid View Swaps

All bidirectional swaps between views are supported:

| Source      | Target      | Axis Permutation | Rotation Axis |
|-------------|-------------|------------------|---------------|
| Axial       | Coronal     | (0, 2, 1)        | Around X      |
| Coronal     | Axial       | (0, 2, 1)        | Around X      |
| Axial       | Sagittal    | (2, 1, 0)        | Around Y      |
| Sagittal    | Axial       | (2, 1, 0)        | Around Y      |
| Coronal     | Sagittal    | (1, 0, 2)        | Around Z      |
| Sagittal    | Coronal     | (1, 0, 2)        | Around Z      |

## Transformation Details

### Axis Permutation

The function reorders axes to align with the target view:

**Example: Axial to Sagittal**
```python
# Original (axial): (X, Y, Z) = (256, 256, 128)
# Permutation: (2, 1, 0) → swap X and Z
# Result (sagittal): (Z, Y, X) = (128, 256, 256)
```

### Rotation

After permutation, a 90-degree clockwise rotation is applied:

```python
# Rotation depends on swap type
rotated = np.rot90(permuted_data, k=1, axes=rotation_axes)
```

### Affine Matrix Update

The affine matrix is updated to maintain correct spatial coordinates:

```python
# Update rotation component
new_affine[:3, :3] = original_affine[:3, :3][new_axes, :]

# Preserve translation
new_affine[:3, 3] = original_affine[:3, 3]
```

This ensures the volume maintains proper spatial registration.

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The input file does not exist                                     |
| `ValueError`         | File is not in `.nii.gz` format                                   |
| `ValueError`         | Input is not a 3D volume                                          |
| `ValueError`         | Invalid source or target view                                      |
| `ValueError`         | Invalid view swap combination                                      |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are accepted
- **3D Volumes Required**: Input must be a 3D NIfTI image
- **Bidirectional**: All view swaps work in both directions
- **Spatial Metadata**: Affine and header are updated appropriately
- **Output Directory**: Automatically created if it doesn't exist
- **No Data Loss**: Transformation is lossless (no interpolation)

## Examples

### Basic View Swap
Convert from axial to sagittal orientation:

```python
from nidataset.volume import swap_nifti_views

swap_nifti_views(
    nii_path="scans/brain_axial.nii.gz",
    output_path="scans/reoriented/",
    source_view="axial",
    target_view="sagittal"
)
# Output: scans/reoriented/brain_axial_swapped_axial_to_sagittal.nii.gz
```

### With Debug Information
Enable verbose output:

```python
swap_nifti_views(
    nii_path="data/scan.nii.gz",
    output_path="data/oriented/",
    source_view="coronal",
    target_view="axial",
    debug=True
)
# Prints:
# Input file: 'data/scan.nii.gz'
# Output path: 'data/oriented/'
# Original shape: (256, 256, 128) | Swapped shape: (256, 128, 256)
# View swapped from coronal to axial
# Swapped NIfTI saved at: data/oriented/scan_swapped_coronal_to_axial.nii.gz
```

### All View Conversions
Convert a volume to all anatomical views:

```python
from nidataset.volume import swap_nifti_views

original_scan = "original/axial_scan.nii.gz"
output_folder = "all_views/"

# Convert to coronal
swap_nifti_views(
    nii_path=original_scan,
    output_path=output_folder,
    source_view="axial",
    target_view="coronal",
    debug=True
)

# Convert to sagittal
swap_nifti_views(
    nii_path=original_scan,
    output_path=output_folder,
    source_view="axial",
    target_view="sagittal",
    debug=True
)

print("Volume available in all three views")
```

### Standardizing Dataset Orientations
Ensure all volumes have the same orientation:

```python
import os
from nidataset.volume import swap_nifti_views

# Dataset with mixed orientations
volumes = {
    'scan_001.nii.gz': 'axial',
    'scan_002.nii.gz': 'coronal',
    'scan_003.nii.gz': 'sagittal'
}

target_orientation = 'axial'
input_folder = "mixed_orientations/"
output_folder = "standardized_axial/"

for filename, current_view in volumes.items():
    if current_view != target_orientation:
        print(f"Converting {filename} from {current_view} to {target_orientation}")
        
        swap_nifti_views(
            nii_path=os.path.join(input_folder, filename),
            output_path=output_folder,
            source_view=current_view,
            target_view=target_orientation,
            debug=True
        )
    else:
        # Already in target orientation, just copy
        import shutil
        shutil.copy(
            os.path.join(input_folder, filename),
            os.path.join(output_folder, filename)
        )

print(f"\nAll volumes standardized to {target_orientation} orientation")
```

### Verifying Transformation
Check that transformation preserves data correctly:

```python
import nibabel as nib
import numpy as np
from nidataset.volume import swap_nifti_views

# Original volume
original = nib.load("verification/scan.nii.gz")
orig_data = original.get_fdata()

print("Original:")
print(f"  Shape: {orig_data.shape}")
print(f"  Data range: [{orig_data.min():.1f}, {orig_data.max():.1f}]")
print(f"  Mean: {orig_data.mean():.1f}")

# Swap view
swap_nifti_views(
    nii_path="verification/scan.nii.gz",
    output_path="verification/swapped/",
    source_view="axial",
    target_view="sagittal",
    debug=True
)

# Load swapped
swapped = nib.load("verification/swapped/scan_swapped_axial_to_sagittal.nii.gz")
swapped_data = swapped.get_fdata()

print("\nSwapped:")
print(f"  Shape: {swapped_data.shape}")
print(f"  Data range: [{swapped_data.min():.1f}, {swapped_data.max():.1f}]")
print(f"  Mean: {swapped_data.mean():.1f}")

# Verify data preservation
print("\nVerification:")
print(f"  Total voxels preserved: {np.prod(orig_data.shape) == np.prod(swapped_data.shape)}")
print(f"  Data range preserved: {orig_data.min() == swapped_data.min() and orig_data.max() == swapped_data.max()}")
print(f"  Mean preserved: {np.isclose(orig_data.mean(), swapped_data.mean())}")
```

### Round-Trip Transformation
Verify bidirectional conversion:

```python
import nibabel as nib
import numpy as np
from nidataset.volume import swap_nifti_views

# Original
original = nib.load("roundtrip/original.nii.gz")
orig_data = original.get_fdata()

# Forward: axial → sagittal
swap_nifti_views(
    nii_path="roundtrip/original.nii.gz",
    output_path="roundtrip/forward/",
    source_view="axial",
    target_view="sagittal",
    debug=True
)

# Backward: sagittal → axial
swap_nifti_views(
    nii_path="roundtrip/forward/original_swapped_axial_to_sagittal.nii.gz",
    output_path="roundtrip/backward/",
    source_view="sagittal",
    target_view="axial",
    debug=True
)

# Load round-trip result
roundtrip = nib.load("roundtrip/backward/original_swapped_axial_to_sagittal_swapped_sagittal_to_axial.nii.gz")
roundtrip_data = roundtrip.get_fdata()

# Verify
print("\nRound-trip Verification:")
print(f"  Original shape: {orig_data.shape}")
print(f"  Round-trip shape: {roundtrip_data.shape}")
print(f"  Shapes match: {orig_data.shape == roundtrip_data.shape}")
print(f"  Data identical: {np.allclose(orig_data, roundtrip_data)}")
```

### Preparing for Multi-View Analysis
Create views for comprehensive visualization:

```python
from nidataset.volume import swap_nifti_views
from nidataset.slices import extract_slices

scan_file = "analysis/brain.nii.gz"

# Generate all three views
views = ["axial", "coronal", "sagittal"]
current_view = "axial"

for target_view in views:
    if target_view != current_view:
        # Swap to target view
        swap_nifti_views(
            nii_path=scan_file,
            output_path=f"multi_view/{target_view}/",
            source_view=current_view,
            target_view=target_view,
            debug=True
        )
        
        # Extract 2D slices from this view
        swapped_file = f"multi_view/{target_view}/brain_swapped_{current_view}_to_{target_view}.nii.gz"
        extract_slices(
            nii_path=swapped_file,
            output_path=f"slices/{target_view}/",
            view=target_view,
            debug=True
        )
    else:
        # Already in correct view
        extract_slices(
            nii_path=scan_file,
            output_path=f"slices/{target_view}/",
            view=target_view,
            debug=True
        )

print("Multi-view analysis prepared")
```

### Batch Orientation Standardization
Process entire dataset:

```python
import os
from nidataset.volume import swap_nifti_views

def standardize_orientations(input_folder, output_folder, metadata, target_view="axial"):
    """
    Standardize all volumes to target orientation.
    
    Args:
        input_folder: Path to input volumes
        output_folder: Path for standardized outputs
        metadata: Dict mapping filenames to their current orientations
        target_view: Desired orientation for all volumes
    """
    os.makedirs(output_folder, exist_ok=True)
    
    converted = 0
    copied = 0
    
    for filename, current_view in metadata.items():
        input_path = os.path.join(input_folder, filename)
        
        if current_view != target_view:
            print(f"Converting {filename}: {current_view} → {target_view}")
            
            swap_nifti_views(
                nii_path=input_path,
                output_path=output_folder,
                source_view=current_view,
                target_view=target_view,
                debug=False
            )
            converted += 1
        else:
            # Already correct orientation
            import shutil
            shutil.copy(input_path, os.path.join(output_folder, filename))
            copied += 1
    
    print(f"\nStandardization complete:")
    print(f"  Converted: {converted}")
    print(f"  Already correct: {copied}")

# Example usage
volume_metadata = {
    'patient_001.nii.gz': 'axial',
    'patient_002.nii.gz': 'coronal',
    'patient_003.nii.gz': 'axial',
    'patient_004.nii.gz': 'sagittal'
}

standardize_orientations(
    "raw_scans/",
    "standardized_scans/",
    volume_metadata,
    target_view="axial"
)
```

### Creating Comparison Visualizations
Generate side-by-side view comparisons:

```python
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nidataset.volume import swap_nifti_views

# Original axial
original = nib.load("comparison/brain_axial.nii.gz")
orig_data = original.get_fdata()

# Convert to sagittal
swap_nifti_views(
    nii_path="comparison/brain_axial.nii.gz",
    output_path="comparison/converted/",
    source_view="axial",
    target_view="sagittal",
    debug=True
)

sagittal = nib.load("comparison/converted/brain_axial_swapped_axial_to_sagittal.nii.gz")
sag_data = sagittal.get_fdata()

# Extract middle slices
axial_slice = orig_data[:, :, orig_data.shape[2]//2]
sagittal_slice = sag_data[sag_data.shape[0]//2, :, :]

# Create comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(axial_slice.T, cmap='gray', origin='lower')
axes[0].set_title(f'Axial View\n{axial_slice.shape}')
axes[0].axis('off')

axes[1].imshow(sagittal_slice.T, cmap='gray', origin='lower')
axes[1].set_title(f'Sagittal View\n{sagittal_slice.shape}')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('view_comparison.png', dpi=150)
print("View comparison saved: view_comparison.png")
```

### Error Handling
Robust processing with validation:

```python
from nidataset.volume import swap_nifti_views

def safe_swap_views(nii_path, output_path, source_view, target_view):
    """Swap views with comprehensive error handling."""
    
    # Validate inputs
    valid_views = ['axial', 'coronal', 'sagittal']
    
    if source_view not in valid_views:
        print(f"Error: Invalid source view '{source_view}'")
        print(f"Must be one of: {valid_views}")
        return False
    
    if target_view not in valid_views:
        print(f"Error: Invalid target view '{target_view}'")
        print(f"Must be one of: {valid_views}")
        return False
    
    if source_view == target_view:
        print("Warning: Source and target views are the same")
        return False
    
    # Attempt swap
    try:
        swap_nifti_views(
            nii_path=nii_path,
            output_path=output_path,
            source_view=source_view,
            target_view=target_view,
            debug=True
        )
        print(f"✓ Successfully swapped {source_view} → {target_view}")
        return True
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        return False
        
    except ValueError as e:
        print(f"✗ Invalid input: {e}")
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

# Use with error handling
safe_swap_views(
    "data/scan.nii.gz",
    "data/converted/",
    "axial",
    "sagittal"
)
```

## Typical Workflow

```python
from nidataset.volume import swap_nifti_views
import nibabel as nib

# 1. Check current orientation
original = nib.load("data/scan.nii.gz")
print(f"Current shape: {original.shape}")
print(f"Current orientation: assumed axial")

# 2. Convert to desired orientation
swap_nifti_views(
    nii_path="data/scan.nii.gz",
    output_path="data/reoriented/",
    source_view="axial",
    target_view="sagittal",
    debug=True
)

# 3. Verify conversion
converted = nib.load("data/reoriented/scan_swapped_axial_to_sagittal.nii.gz")
print(f"\nConverted shape: {converted.shape}")

# 4. Use reoriented volume for:
# - View-specific analysis
# - Consistent dataset orientation
# - Multi-view visualization
# - Orientation-dependent processing
```