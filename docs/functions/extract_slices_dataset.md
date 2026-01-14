---
title: extract_slices_dataset
parent: Package Functions
nav_order: 11
---

# `extract_slices_dataset`

Batch extract 2D slices from all 3D volumes in a dataset folder, with flexible organization options and optional statistics tracking.

```python
extract_slices_dataset(
    nii_folder: str,
    output_path: str,
    view: str = "axial",
    saving_mode: str = "case",
    target_size: Optional[Tuple[int, int]] = None,
    pad_value: float = 0.0,
    save_stats: bool = False
) -> None
```

## Overview

This function processes all NIfTI volumes in a dataset folder and extracts 2D slices along a specified anatomical plane. It provides flexible control over:

- **Anatomical view**: Extract axial, coronal, or sagittal slices
- **Organization**: Group by case or by view
- **Padding**: Optional uniform sizing with customizable padding values
- **Statistics**: Optional tracking of slice counts per volume

The function is designed to prepare large medical imaging datasets for 2D deep learning models, visualization, or quality control workflows.

## Parameters

| Name          | Type                        | Default    | Description                                                                                          |
|---------------|-----------------------------|------------|------------------------------------------------------------------------------------------------------|
| `nii_folder`  | `str`                       | *required* | Path to the directory containing NIfTI volumes in `.nii.gz` format.                                 |
| `output_path` | `str`                       | *required* | Root directory where extracted slices will be saved.                                                |
| `view`        | `str`                       | `"axial"`  | Anatomical view for extraction: `"axial"`, `"coronal"`, or `"sagittal"`.                           |
| `saving_mode` | `str`                       | `"case"`   | Organization mode: `"case"` (folder per file) or `"view"` (shared folder).                         |
| `target_size` | `Optional[Tuple[int, int]]` | `None`     | Target dimensions (height, width) for padding. If `None`, slices saved at original size.           |
| `pad_value`   | `float`                     | `0.0`      | Value used for padding when `target_size` is specified.                                             |
| `save_stats`  | `bool`                      | `False`    | If `True`, saves slice count statistics as `<view>_slices_stats.csv`.                              |

## Returns

`None` – The function saves TIFF images to disk.

## Output Organization

### Saving Modes

#### Case Mode (`saving_mode="case"`)
Creates a separate folder for each volume:
```
output_path/
├── patient_001/
│   └── axial/
│       ├── patient_001_axial_000.tif
│       ├── patient_001_axial_001.tif
│       └── ...
├── patient_002/
│   └── axial/
│       └── ...
```

#### View Mode (`saving_mode="view"`)
Groups all slices in a single view folder:
```
output_path/
└── axial/
    ├── patient_001_axial_000.tif
    ├── patient_001_axial_001.tif
    ├── patient_002_axial_000.tif
    └── ...
```

### Filename Pattern
Each slice is saved with the format:
```
<PREFIX>_<VIEW>_<SLICE_NUMBER>.tif
```
where:
- `<PREFIX>`: Original filename without `.nii.gz`
- `<VIEW>`: The anatomical view (`axial`, `coronal`, or `sagittal`)
- `<SLICE_NUMBER>`: Zero-padded slice index (e.g., `000`, `001`, `002`)

**Example**: `patient_042_axial_015.tif`

## Anatomical Views

The `view` parameter determines which axis to extract along:

| View         | Extraction Axis | Description                    | Typical Use Case              |
|--------------|-----------------|--------------------------------|-------------------------------|
| `"axial"`    | Z-axis          | Horizontal slices (top-down)   | Brain imaging, chest scans    |
| `"coronal"`  | Y-axis          | Frontal slices (front-back)    | Spine imaging, full body      |
| `"sagittal"` | X-axis          | Lateral slices (left-right)    | Brain hemispheres, symmetry   |

## Padding and Sizing

### Without Padding (`target_size=None`)
Slices are saved at their original dimensions, which may vary across the dataset.

### With Padding (`target_size=(H, W)`)
All slices are padded symmetrically to the specified dimensions:
- Padding is applied equally on all sides (centered)
- If odd padding is needed, the extra pixel goes to the right/bottom
- Padding uses the value specified in `pad_value`

**Example**:
```python
# Original slice: 384×384
# target_size=(512, 512)
# Padding added: 64 pixels on each side
# Result: 512×512 slice with centered content
```

**Important**: When using padding with annotations, ensure `extract_annotations_dataset` uses the same `target_size` for coordinate alignment.

## Statistics File

When `save_stats=True`, a CSV file is created with slice counts:

| Column        | Description                        |
|---------------|------------------------------------|
| `FILENAME`    | Volume filename                    |
| `NUM_SLICES`  | Number of slices in the volume     |
| `TOTAL_SLICES`| Sum across all files (last row)    |

The file is named `<view>_slices_stats.csv` and saved in `output_path`.

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |
| `ValueError`         | Invalid `view` or `saving_mode`                                    |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **Output Format**: Slices are saved as TIFF (`.tif`) images
- **Progress Display**: Shows progress bar with current file being processed
- **Error Handling**: Files that fail are skipped with error messages
- **Output Directory**: Automatically created if it doesn't exist
- **Slice Ordering**: Slices are numbered sequentially from 0

## Examples

### Basic Usage - Axial Slices
Extract axial slices organized by case:

```python
from nidataset.slices import extract_slices_dataset

extract_slices_dataset(
    nii_folder="dataset/scans/",
    output_path="extracted/images/",
    view="axial",
    saving_mode="case"
)
# Creates: extracted/images/case_001/axial/case_001_axial_000.tif, ...
```

### With Statistics Tracking
Enable statistics for dataset overview:

```python
extract_slices_dataset(
    nii_folder="data/volumes/",
    output_path="data/slices/",
    view="coronal",
    saving_mode="view",
    save_stats=True
)
# Creates: data/slices/coronal_slices_stats.csv
```

### Uniform Sizing with Padding
Extract and pad all slices to 512×512:

```python
extract_slices_dataset(
    nii_folder="raw_data/",
    output_path="preprocessed/padded_slices/",
    view="axial",
    saving_mode="case",
    target_size=(512, 512),
    pad_value=0.0,
    save_stats=True
)
# All slices padded to 512×512 with zeros
```

### Custom Padding Value
Use a different padding value (e.g., minimum intensity):

```python
extract_slices_dataset(
    nii_folder="scans/",
    output_path="slices/",
    view="sagittal",
    saving_mode="view",
    target_size=(256, 256),
    pad_value=-1024.0,  # Hounsfield unit for air in CT
    save_stats=True
)
```

### Multi-View Extraction
Extract slices from all three anatomical views:

```python
from nidataset.slices import extract_slices_dataset

views = ["axial", "coronal", "sagittal"]
base_path = "multi_view_dataset/"

for view in views:
    print(f"Extracting {view} slices...")
    extract_slices_dataset(
        nii_folder="volumes/",
        output_path=base_path,
        view=view,
        saving_mode="view",
        target_size=(512, 512),
        save_stats=True
    )
# Creates separate folders for each view with statistics
```

### Complete Training Data Pipeline
Extract images and annotations together:

```python
from nidataset.slices import extract_slices_dataset, extract_annotations_dataset

# Configuration
input_scans = "data/scans/"
input_masks = "data/masks/"
output_base = "training_data/"
view_type = "axial"
uniform_size = (512, 512)

# Extract images with padding
extract_slices_dataset(
    nii_folder=input_scans,
    output_path=f"{output_base}/images/",
    view=view_type,
    saving_mode="case",
    target_size=uniform_size,
    pad_value=0.0,
    save_stats=True
)

# Extract annotations with matching padding adjustment
extract_annotations_dataset(
    nii_folder=input_masks,
    output_path=f"{output_base}/labels/",
    view=view_type,
    saving_mode="case",
    extraction_mode="slice",
    data_mode="box",
    target_size=uniform_size,  # Must match image extraction
    save_stats=True
)

print("Training data prepared with aligned images and annotations")
```

### Analyzing Statistics
Review slice distribution across dataset:

```python
import pandas as pd
from nidataset.slices import extract_slices_dataset

# Extract slices with statistics
extract_slices_dataset(
    nii_folder="dataset/",
    output_path="slices/",
    view="axial",
    saving_mode="case",
    save_stats=True
)

# Load and analyze statistics
stats = pd.read_csv("slices/axial_slices_stats.csv")

# Remove total row for per-file analysis
per_file = stats[stats['FILENAME'] != 'TOTAL_SLICES'].copy()
per_file['NUM_SLICES'] = pd.to_numeric(per_file['NUM_SLICES'])

print("Slice Statistics:")
print(f"  Total files: {len(per_file)}")
print(f"  Total slices: {per_file['NUM_SLICES'].sum()}")
print(f"  Average slices per volume: {per_file['NUM_SLICES'].mean():.1f}")
print(f"  Min slices: {per_file['NUM_SLICES'].min()}")
print(f"  Max slices: {per_file['NUM_SLICES'].max()}")

# Identify outliers
mean_slices = per_file['NUM_SLICES'].mean()
std_slices = per_file['NUM_SLICES'].std()

outliers = per_file[
    (per_file['NUM_SLICES'] < mean_slices - 2*std_slices) |
    (per_file['NUM_SLICES'] > mean_slices + 2*std_slices)
]

if not outliers.empty:
    print(f"\nOutliers detected ({len(outliers)} files):")
    print(outliers[['FILENAME', 'NUM_SLICES']])
```

### Quality Control Workflow
Verify extraction and check for issues:

```python
import os
from PIL import Image
from nidataset.slices import extract_slices_dataset

# Extract slices
extract_slices_dataset(
    nii_folder="qa/volumes/",
    output_path="qa/extracted/",
    view="axial",
    saving_mode="case",
    target_size=(512, 512),
    save_stats=True
)

# Check a sample case
sample_case = "qa/extracted/case_001/axial/"
slice_files = sorted([f for f in os.listdir(sample_case) if f.endswith('.tif')])

print(f"Sample case has {len(slice_files)} slices")

# Load and verify first slice
first_slice = Image.open(os.path.join(sample_case, slice_files[0]))
print(f"Slice dimensions: {first_slice.size}")
print(f"Slice mode: {first_slice.mode}")

# Check for empty slices
for slice_file in slice_files[:10]:  # Check first 10
    img = Image.open(os.path.join(sample_case, slice_file))
    img_array = np.array(img)
    if img_array.max() == 0:
        print(f"Warning: Empty slice detected - {slice_file}")
```

### Batch Processing Different Views
Process multiple datasets with different configurations:

```python
from nidataset.slices import extract_slices_dataset

datasets = {
    'brain': {'folder': 'data/brain_scans/', 'view': 'axial', 'size': (256, 256)},
    'spine': {'folder': 'data/spine_scans/', 'view': 'sagittal', 'size': (512, 256)},
    'chest': {'folder': 'data/chest_scans/', 'view': 'coronal', 'size': (512, 512)}
}

for name, config in datasets.items():
    print(f"Processing {name} dataset...")
    extract_slices_dataset(
        nii_folder=config['folder'],
        output_path=f"extracted/{name}/",
        view=config['view'],
        saving_mode="case",
        target_size=config['size'],
        save_stats=True
    )
```

### Creating Train/Val/Test Splits
Extract and organize slices for model training:

```python
import os
import shutil
from nidataset.slices import extract_slices_dataset

# First, extract all slices
extract_slices_dataset(
    nii_folder="all_scans/",
    output_path="all_slices/",
    view="axial",
    saving_mode="case",
    target_size=(512, 512),
    save_stats=True
)

# Define splits (example case IDs)
splits = {
    'train': ['case_001', 'case_002', 'case_003'],
    'val': ['case_004'],
    'test': ['case_005']
}

# Organize into split folders
for split_name, case_ids in splits.items():
    split_folder = f"dataset/{split_name}/"
    os.makedirs(split_folder, exist_ok=True)
    
    for case_id in case_ids:
        src = f"all_slices/{case_id}/"
        dst = f"{split_folder}/{case_id}/"
        if os.path.exists(src):
            shutil.copytree(src, dst)
    
    print(f"{split_name}: {len(case_ids)} cases")
```

### Comparing Slice Counts Across Views
Extract and compare all anatomical views:

```python
import pandas as pd
from nidataset.slices import extract_slices_dataset

views = ["axial", "coronal", "sagittal"]
comparison = {}

for view in views:
    extract_slices_dataset(
        nii_folder="volumes/",
        output_path=f"comparison/{view}/",
        view=view,
        saving_mode="view",
        save_stats=True
    )
    
    stats = pd.read_csv(f"comparison/{view}/{view}_slices_stats.csv")
    per_file = stats[stats['FILENAME'] != 'TOTAL_SLICES']
    comparison[view] = {
        'total': per_file['NUM_SLICES'].sum(),
        'mean': per_file['NUM_SLICES'].mean(),
        'std': per_file['NUM_SLICES'].std()
    }

# Display comparison
comp_df = pd.DataFrame(comparison).T
print("\nSlice Count Comparison Across Views:")
print(comp_df)
```

## Typical Workflow

```python
from nidataset.slices import extract_slices_dataset
import pandas as pd

# 1. Define paths and parameters
volume_folder = "dataset/medical_scans/"
slice_output = "dataset/extracted_slices/"
anatomical_view = "axial"
standard_size = (512, 512)

# 2. Extract slices with uniform sizing
extract_slices_dataset(
    nii_folder=volume_folder,
    output_path=slice_output,
    view=anatomical_view,
    saving_mode="case",
    target_size=standard_size,
    pad_value=0.0,
    save_stats=True
)

# 3. Review statistics
stats = pd.read_csv(f"{slice_output}/{anatomical_view}_slices_stats.csv")
print(f"Extracted {stats.iloc[-1]['TOTAL_SLICES']} total slices")

# 4. Use extracted slices for:
# - Training 2D deep learning models
# - Data augmentation pipelines
# - Visualization and quality control
# - Dataset analysis and validation
```