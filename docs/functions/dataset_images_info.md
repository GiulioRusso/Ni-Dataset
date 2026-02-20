---
title: dataset_images_info
parent: Package Functions
nav_order: 10
---

# `dataset_images_info`

Extract comprehensive metadata from all NIfTI volumes in a folder and save the summary as a CSV file for dataset analysis and quality control.

```python
dataset_images_info(
    nii_folder: str,
    output_path: str
) -> None
```

## Overview

This function generates a detailed metadata summary for medical imaging datasets. It extracts key properties from each NIfTI file including spatial dimensions, voxel sizes, intensity statistics, and tissue volume measurements. The resulting CSV is useful for:

- Dataset quality control and validation
- Identifying outliers or corrupted files
- Understanding data distribution before preprocessing
- Documentation and reproducibility
- Comparing multiple datasets

All metadata is saved to `dataset_images_info.csv` in the specified output directory.

## Parameters

| Name          | Type  | Default    | Description                                                                              |
|---------------|-------|------------|------------------------------------------------------------------------------------------|
| `nii_folder`  | `str` | *required* | Path to the directory containing NIfTI volumes in `.nii.gz` format.                     |
| `output_path` | `str` | *required* | Directory where the CSV file will be saved. Created automatically if it doesn't exist.  |

## Returns

`None` – The function saves results to disk.

## Output File

### CSV Structure
The function creates `dataset_images_info.csv` with the following columns:

| Column                | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `FILENAME`            | Name of the NIfTI file                                                  |
| `SHAPE (X, Y, Z)`     | Image dimensions in voxels as `(width, height, depth)`                  |
| `VOXEL SIZE (mm)`     | Physical size of each voxel as `(x_size, y_size, z_size)`              |
| `DATA TYPE`           | NumPy data type (e.g., `float64`, `int16`, `uint8`)                     |
| `MIN VALUE`           | Minimum intensity value in the volume                                    |
| `MAX VALUE`           | Maximum intensity value in the volume                                    |
| `BRAIN VOXELS`        | Count of non-zero voxels (tissue volume)                                |
| `BRAIN VOLUME (mm³)`  | Physical volume of non-zero voxels in cubic millimeters                 |
| `BBOX MIN (X, Y, Z)`  | Minimum coordinates of the bounding box around non-zero voxels          |
| `BBOX MAX (X, Y, Z)`  | Maximum coordinates of the bounding box around non-zero voxels          |

### Example CSV Output
```
FILENAME,SHAPE (X, Y, Z),VOXEL SIZE (mm),DATA TYPE,MIN VALUE,MAX VALUE,BRAIN VOXELS,BRAIN VOLUME (mm³),BBOX MIN (X, Y, Z),BBOX MAX (X, Y, Z)
scan_001.nii.gz,"(512, 512, 300)","(0.5, 0.5, 1.0)",float64,0.0,4095.0,45678900,22839450.0,"[50, 60, 20]","[462, 452, 280]"
scan_002.nii.gz,"(256, 256, 150)","(1.0, 1.0, 1.5)",float32,-1024.0,3071.0,12456780,18685170.0,"[30, 40, 15]","[226, 216, 135]"
```

## Metadata Details

### Shape and Voxel Size
- **Shape** represents the number of voxels in each dimension
- **Voxel size** indicates the physical spacing between voxels in millimeters
- Together, they determine the physical dimensions: `physical_size = shape × voxel_size`

### Intensity Range
- **MIN VALUE** and **MAX VALUE** show the intensity range in the volume
- Useful for detecting preprocessing issues or unexpected value ranges
- Different modalities have different typical ranges (e.g., Hounsfield units for CT)

### Tissue Volume Metrics
- **BRAIN VOXELS**: Count of non-zero voxels, representing tissue or contrast-enhanced regions
- **BRAIN VOLUME**: Physical volume calculated as `non_zero_count × voxel_x × voxel_y × voxel_z`
- Note: "BRAIN" terminology is used generically for non-zero regions, applicable to any tissue type

### Bounding Box
- Minimum and maximum coordinates defining the smallest box containing all non-zero voxels
- Useful for automatic cropping and region of interest extraction
- Coordinates are in voxel space (0-indexed)

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `nii_folder` does not exist or contains no `.nii.gz` files    |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **Progress Display**: Shows a progress bar during metadata extraction
- **Error Handling**: Files that fail to process are skipped with error messages
- **Output Directory**: Automatically created if it doesn't exist
- **Non-zero Definition**: Voxels with intensity > 0 are considered tissue

## Examples

### Basic Usage
Extract metadata for all volumes in a folder:

```python
from nidataset.utility import dataset_images_info

dataset_images_info(
    nii_folder="dataset/scans/",
    output_path="dataset/metadata/"
)
# Creates: dataset/metadata/dataset_images_info.csv
```

### Quality Control Analysis
Load and analyze the metadata to identify outliers:

```python
import pandas as pd
from nidataset.utility import dataset_images_info

# Extract metadata
dataset_images_info(
    nii_folder="data/raw_scans/",
    output_path="data/qa/"
)

# Load and analyze
df = pd.read_csv("data/qa/dataset_images_info.csv")

# Check for dimension consistency
print("Unique shapes in dataset:")
print(df['SHAPE (X, Y, Z)'].value_counts())

# Check for unusual voxel sizes
print("\nVoxel size distribution:")
print(df['VOXEL SIZE (mm)'].value_counts())

# Identify volumes with unusual intensity ranges
print("\nIntensity range summary:")
print(df[['MIN VALUE', 'MAX VALUE']].describe())

# Find potentially corrupted files (very small volumes)
min_expected_volume = 100000  # mm³
suspicious = df[df['BRAIN VOLUME (mm³)'] < min_expected_volume]
if not suspicious.empty:
    print(f"\nWarning: {len(suspicious)} files with unusually small volumes:")
    print(suspicious[['FILENAME', 'BRAIN VOLUME (mm³)']])
```

### Dataset Comparison
Compare metadata across multiple datasets:

```python
import pandas as pd
from nidataset.utility import dataset_images_info

# Extract metadata for multiple datasets
datasets = {
    'Training': 'data/train/',
    'Validation': 'data/val/',
    'Testing': 'data/test/'
}

for name, folder in datasets.items():
    dataset_images_info(
        nii_folder=folder,
        output_path=f"metadata/{name}/"
    )

# Compare datasets
for name in datasets.keys():
    df = pd.read_csv(f"metadata/{name}/dataset_images_info.csv")
    print(f"\n{name} Dataset:")
    print(f"  Files: {len(df)}")
    print(f"  Avg volume: {df['BRAIN VOLUME (mm³)'].mean():.0f} mm³")
    print(f"  Shape consistency: {df['SHAPE (X, Y, Z)'].nunique()} unique shapes")
```

### Preprocessing Planning
Use metadata to determine appropriate preprocessing parameters:

```python
import pandas as pd
import ast
from nidataset.utility import dataset_images_info

# Extract metadata
dataset_images_info(
    nii_folder="data/original/",
    output_path="data/analysis/"
)

# Analyze bounding boxes to determine crop size
df = pd.read_csv("data/analysis/dataset_images_info.csv")

# Calculate bounding box dimensions
bbox_sizes = []
for idx, row in df.iterrows():
    bbox_min = ast.literal_eval(row['BBOX MIN (X, Y, Z)'])
    bbox_max = ast.literal_eval(row['BBOX MAX (X, Y, Z)'])
    size = [bbox_max[i] - bbox_min[i] for i in range(3)]
    bbox_sizes.append(size)

bbox_df = pd.DataFrame(bbox_sizes, columns=['X', 'Y', 'Z'])

print("Bounding box size statistics:")
print(bbox_df.describe())

# Recommend target shape (95th percentile)
recommended_shape = tuple(bbox_df.quantile(0.95).astype(int).values)
print(f"\nRecommended target shape for crop_and_pad: {recommended_shape}")
```

### Data Type Verification
Check if data types are consistent across the dataset:

```python
import pandas as pd
from nidataset.utility import dataset_images_info

dataset_images_info(
    nii_folder="data/scans/",
    output_path="data/info/"
)

df = pd.read_csv("data/info/dataset_images_info.csv")

# Check data type consistency
print("Data types in dataset:")
print(df['DATA TYPE'].value_counts())

# Identify files with unexpected data types
expected_dtype = 'float64'
unexpected = df[df['DATA TYPE'] != expected_dtype]
if not unexpected.empty:
    print(f"\nWarning: {len(unexpected)} files with unexpected data type:")
    print(unexpected[['FILENAME', 'DATA TYPE']])
```

### Export Summary Report
Generate a human-readable summary report:

```python
import pandas as pd
from nidataset.utility import dataset_images_info

# Extract metadata
dataset_images_info(
    nii_folder="dataset/images/",
    output_path="dataset/reports/"
)

# Load and create summary
df = pd.read_csv("dataset/reports/dataset_images_info.csv")

summary = f"""
Dataset Summary Report
=====================

Total Files: {len(df)}

Dimensions:
  - Shapes: {df['SHAPE (X, Y, Z)'].nunique()} unique
  - Most common: {df['SHAPE (X, Y, Z)'].mode()[0]}

Voxel Spacing:
  - Voxel sizes: {df['VOXEL SIZE (mm)'].nunique()} unique
  - Most common: {df['VOXEL SIZE (mm)'].mode()[0]}

Intensity:
  - Global min: {df['MIN VALUE'].min()}
  - Global max: {df['MAX VALUE'].max()}

Volume:
  - Mean brain volume: {df['BRAIN VOLUME (mm³)'].mean():.0f} mm³
  - Std brain volume: {df['BRAIN VOLUME (mm³)'].std():.0f} mm³
  - Range: [{df['BRAIN VOLUME (mm³)'].min():.0f}, {df['BRAIN VOLUME (mm³)'].max():.0f}] mm³
"""

print(summary)

# Save report
with open("dataset/reports/summary.txt", "w") as f:
    f.write(summary)
```

## Typical Workflow

```python
from nidataset.utility import dataset_images_info
import pandas as pd

# 1. Extract metadata for your dataset
dataset_images_info(
    nii_folder="data/medical_scans/",
    output_path="data/metadata/"
)

# 2. Load the results
df = pd.read_csv("data/metadata/dataset_images_info.csv")

# 3. Perform quality checks
print(f"Dataset contains {len(df)} volumes")
print(f"Dimension consistency: {df['SHAPE (X, Y, Z)'].nunique()} unique shapes")

# 4. Use metadata to inform preprocessing decisions
# - Determine appropriate crop sizes
# - Identify files needing special handling
# - Verify data type consistency
# - Check for outliers or corrupted files
```