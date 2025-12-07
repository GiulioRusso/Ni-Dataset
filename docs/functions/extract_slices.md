---
title: extract_slices
parent: Package Functions
nav_order: 11
---

# `extract_slices`

Extract 2D slices from a 3D NIfTI volume along a specified anatomical plane and save them as TIFF images.

```python
extract_slices(
    nii_path: str,
    output_path: str,
    view: str = "axial",
    target_size: Optional[Tuple[int, int]] = None,
    pad_value: float = 0.0,
    debug: bool = False
) -> None
```

## Overview

This function converts a 3D medical imaging volume into a series of 2D slices along a chosen anatomical plane. Each slice is saved as a separate TIFF image with consistent naming, making it suitable for:

- Preparing data for 2D deep learning models
- Visual inspection and quality control
- Creating training datasets for slice-based analysis
- Exporting data for visualization tools

The function supports optional padding to create uniformly sized slices across datasets with varying dimensions.

## Parameters

| Name          | Type                        | Default    | Description                                                                                          |
|---------------|-----------------------------|------------|------------------------------------------------------------------------------------------------------|
| `nii_path`    | `str`                       | *required* | Path to the input NIfTI volume in `.nii.gz` format.                                                 |
| `output_path` | `str`                       | *required* | Directory where extracted slices will be saved. Created automatically if it doesn't exist.          |
| `view`        | `str`                       | `"axial"`  | Anatomical view for extraction: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `target_size` | `Optional[Tuple[int, int]]` | `None`     | Target dimensions (height, width) for padding. If `None`, slices saved at original size.           |
| `pad_value`   | `float`                     | `0.0`      | Value used for padding when `target_size` is specified.                                             |
| `debug`       | `bool`                      | `False`    | If `True`, prints extraction details including slice count and padding information.                 |

## Returns

`None` – The function saves TIFF images to disk.

## Output Files

Each slice is saved with the filename pattern:
```
<PREFIX>_<VIEW>_<SLICE_NUMBER>.tif
```

where:
- `<PREFIX>`: Original filename without the `.nii.gz` extension
- `<VIEW>`: The anatomical view (`axial`, `coronal`, or `sagittal`)
- `<SLICE_NUMBER>`: Zero-padded slice index (e.g., `000`, `001`, `002`)

**Example**: Input `patient_042.nii.gz` → Outputs `patient_042_axial_000.tif`, `patient_042_axial_001.tif`, etc.

## Anatomical Views

The `view` parameter determines which anatomical plane to extract:

| View         | Extraction Axis | Description                    | Common Applications                |
|--------------|-----------------|--------------------------------|------------------------------------|
| `"axial"`    | Z-axis          | Horizontal slices (top-down)   | Brain, chest, abdominal imaging    |
| `"coronal"`  | Y-axis          | Frontal slices (front-back)    | Spine, sinus, full body scans      |
| `"sagittal"` | X-axis          | Lateral slices (left-right)    | Brain hemispheres, symmetry checks |

## Padding Behavior

### Without Padding (`target_size=None`)
Slices are saved at their original dimensions. This is suitable when all volumes in your dataset have the same slice dimensions.

### With Padding (`target_size=(H, W)`)
Slices are padded symmetrically to match the specified dimensions:

**Padding Distribution**:
- Padding is split equally between top/bottom and left/right
- When padding is odd, the extra pixel goes to the right/bottom
- Padding uses the value specified in `pad_value`

**Example**:
```python
# Original slice: 384×384
# target_size=(512, 512)
# Padding needed: 128 pixels total
# Distribution: 64 pixels on each side
# Result: 512×512 slice with centered content
```

**Validation**: The function raises an error if `target_size` is smaller than the original slice dimensions.

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The input file does not exist                                     |
| `ValueError`         | File is not in `.nii.gz` format                                   |
| `ValueError`         | File is not 3D or has zero slices                                 |
| `ValueError`         | Invalid `view` parameter                                          |
| `ValueError`         | `target_size` is smaller than slice dimensions                    |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are accepted
- **3D Volumes Required**: Input must be a 3D NIfTI image
- **Output Format**: Slices are saved as TIFF (`.tif`) images
- **Output Directory**: Automatically created if it doesn't exist
- **Progress Display**: Shows progress bar during extraction
- **Slice Numbering**: Slices are numbered from 0 to (num_slices - 1)

## Examples

### Basic Usage
Extract axial slices at original size:

```python
from nidataset.slices import extract_slices

extract_slices(
    nii_path="scans/patient_001.nii.gz",
    output_path="slices/patient_001/",
    view="axial"
)
# Output: slices/patient_001/patient_001_axial_000.tif, ...
```

### With Debug Information
Enable verbose output to see extraction details:

```python
extract_slices(
    nii_path="volumes/brain_scan.nii.gz",
    output_path="extracted/brain/",
    view="coronal",
    debug=True
)
# Prints:
# Input file: 'volumes/brain_scan.nii.gz'
# Output path: 'extracted/brain/'
# No padding applied
# Total coronal slices extracted: 256
```

### Uniform Sizing with Padding
Pad all slices to a standard size:

```python
extract_slices(
    nii_path="data/scan.nii.gz",
    output_path="data/padded_slices/",
    view="axial",
    target_size=(512, 512),
    pad_value=0.0,
    debug=True
)
# All slices padded to 512×512 with zeros
```

### Custom Padding Value
Use a specific padding value (e.g., for CT scans):

```python
extract_slices(
    nii_path="ct_scans/chest.nii.gz",
    output_path="ct_slices/",
    view="axial",
    target_size=(512, 512),
    pad_value=-1024.0,  # Hounsfield unit for air
    debug=True
)
```

### All Anatomical Views
Extract slices from all three views:

```python
from nidataset.slices import extract_slices

scan_file = "volumes/patient_042.nii.gz"
views = ["axial", "coronal", "sagittal"]

for view in views:
    extract_slices(
        nii_path=scan_file,
        output_path=f"multi_view/{view}/",
        view=view,
        target_size=(512, 512),
        debug=True
    )
# Creates separate folders for each anatomical view
```

### Preparing Training Data
Extract slices with matching annotations:

```python
from nidataset.slices import extract_slices, extract_annotations

scan_file = "data/scan_001.nii.gz"
mask_file = "data/mask_001.nii.gz"
uniform_size = (512, 512)

# Extract image slices with padding
extract_slices(
    nii_path=scan_file,
    output_path="training/images/",
    view="axial",
    target_size=uniform_size,
    pad_value=0.0,
    debug=True
)

# Extract annotations with matching padding adjustment
extract_annotations(
    nii_path=mask_file,
    output_path="training/labels/",
    view="axial",
    saving_mode="slice",
    data_mode="box",
    target_size=uniform_size,  # Must match image extraction
    debug=True
)

print("Training data prepared with aligned images and annotations")
```

### Quality Control - Visual Inspection
Extract slices for manual review:

```python
import os
from PIL import Image
from nidataset.slices import extract_slices

# Extract slices
extract_slices(
    nii_path="qa/suspicious_scan.nii.gz",
    output_path="qa/review/",
    view="axial",
    debug=True
)

# Load and inspect first and last slices
slices = sorted([f for f in os.listdir("qa/review/") if f.endswith('.tif')])
first = Image.open(f"qa/review/{slices[0]}")
last = Image.open(f"qa/review/{slices[-1]}")

print(f"Total slices: {len(slices)}")
print(f"First slice size: {first.size}")
print(f"Last slice size: {last.size}")

# Open slices for visual inspection
first.show()
last.show()
```

### Batch Processing Different Views
Process multiple files with different orientations:

```python
from nidataset.slices import extract_slices

files = [
    {'path': 'scans/brain.nii.gz', 'view': 'axial', 'size': (256, 256)},
    {'path': 'scans/spine.nii.gz', 'view': 'sagittal', 'size': (512, 256)},
    {'path': 'scans/chest.nii.gz', 'view': 'coronal', 'size': (512, 512)}
]

for config in files:
    filename = os.path.basename(config['path']).replace('.nii.gz', '')
    extract_slices(
        nii_path=config['path'],
        output_path=f"extracted/{filename}/",
        view=config['view'],
        target_size=config['size'],
        debug=True
    )
```

### Handling Variable Dimensions
Determine appropriate padding size from dataset:

```python
import nibabel as nib
import os
from nidataset.slices import extract_slices

# Scan all files to find maximum dimensions
scan_folder = "raw_scans/"
max_h, max_w = 0, 0

for filename in os.listdir(scan_folder):
    if filename.endswith('.nii.gz'):
        img = nib.load(os.path.join(scan_folder, filename))
        data = img.get_fdata()
        h, w = data.shape[0], data.shape[1]  # For axial view
        max_h = max(max_h, h)
        max_w = max(max_w, w)

# Round up to nearest power of 2 or multiple of 16
target_h = ((max_h + 15) // 16) * 16
target_w = ((max_w + 15) // 16) * 16

print(f"Dataset max dimensions: {max_h}×{max_w}")
print(f"Using target size: {target_h}×{target_w}")

# Extract with appropriate padding
for filename in os.listdir(scan_folder):
    if filename.endswith('.nii.gz'):
        extract_slices(
            nii_path=os.path.join(scan_folder, filename),
            output_path="padded_slices/",
            view="axial",
            target_size=(target_h, target_w),
            debug=True
        )
```

### Creating Subset of Slices
Extract only specific slices (requires manual filtering):

```python
from nidataset.slices import extract_slices
import os

# First extract all slices
extract_slices(
    nii_path="full_scan.nii.gz",
    output_path="temp_slices/",
    view="axial"
)

# Keep only slices with annotations (example: slices 50-150)
keep_range = range(50, 151)
all_slices = sorted([f for f in os.listdir("temp_slices/") if f.endswith('.tif')])

os.makedirs("filtered_slices/", exist_ok=True)
for slice_file in all_slices:
    slice_idx = int(slice_file.split('_')[-1].replace('.tif', ''))
    if slice_idx in keep_range:
        shutil.copy(
            f"temp_slices/{slice_file}",
            f"filtered_slices/{slice_file}"
        )

print(f"Kept {len(keep_range)} slices from total {len(all_slices)}")
```

### Verifying Padding
Check padding application:

```python
import numpy as np
from PIL import Image
from nidataset.slices import extract_slices

# Extract with padding
extract_slices(
    nii_path="test_scan.nii.gz",
    output_path="padded_test/",
    view="axial",
    target_size=(512, 512),
    pad_value=-100.0,
    debug=True
)

# Load a slice and verify padding
sample = Image.open("padded_test/test_scan_axial_050.tif")
sample_array = np.array(sample)

print(f"Slice shape: {sample_array.shape}")
print(f"Min value: {sample_array.min()}")
print(f"Max value: {sample_array.max()}")

# Check if padding value is present
if -100.0 in sample_array:
    print("Padding detected in slice")
    padding_mask = sample_array == -100.0
    print(f"Padded pixels: {padding_mask.sum()}")
```

### Integration with Visualization Pipeline
Extract and prepare for viewer:

```python
from nidataset.slices import extract_slices
import os

# Extract slices
extract_slices(
    nii_path="visualization/source.nii.gz",
    output_path="visualization/slices/",
    view="axial",
    target_size=(512, 512),
    debug=True
)

# Create image sequence for video or animation
slice_files = sorted([
    f for f in os.listdir("visualization/slices/") 
    if f.endswith('.tif')
])

print(f"Created {len(slice_files)} slices ready for visualization")
print(f"Use these for: video creation, web viewer, or animation")
```

## Typical Workflow

```python
from nidataset.slices import extract_slices
import nibabel as nib

# 1. Define input and output
scan_file = "data/patient_scan.nii.gz"
output_folder = "data/extracted_slices/"
view_type = "axial"

# 2. Check original dimensions (optional)
img = nib.load(scan_file)
data = img.get_fdata()
print(f"Original volume shape: {data.shape}")

# 3. Extract slices with appropriate settings
extract_slices(
    nii_path=scan_file,
    output_path=output_folder,
    view=view_type,
    target_size=(512, 512),
    pad_value=0.0,
    debug=True
)

# 4. Use extracted slices for:
# - Training 2D neural networks
# - Visual quality control
# - Dataset analysis
# - Annotation tools
```