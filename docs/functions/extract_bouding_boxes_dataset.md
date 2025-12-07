---
title: extract_bounding_boxes_dataset
parent: Package Functions
nav_order: 19
---

# `extract_bounding_boxes_dataset`

Batch process segmentation masks to extract 3D bounding boxes around connected components, filtering by volume threshold and saving results as NIfTI files.

```python
extract_bounding_boxes_dataset(
    mask_folder: str,
    output_path: str,
    voxel_size: tuple = (3.0, 3.0, 3.0),
    volume_threshold: float = 1000.0,
    mask_value: int = 1,
    save_stats: bool = True,
    debug: bool = False
) -> None
```

## Overview

This function processes all segmentation masks in a dataset folder to identify and extract bounding boxes around anatomical structures or regions of interest. It uses connected component analysis to detect separate instances and filters them by physical volume to exclude noise or artifacts.

Key features include:
- Volume-based filtering to remove small components
- Physical space calculations using voxel dimensions
- Per-case bounding box visualization
- Optional statistics tracking across the dataset

Each processed mask generates a new NIfTI file containing only the bounding boxes, which can be used for visualization, validation, or further processing.

## Parameters

| Name               | Type    | Default           | Description                                                                                          |
|--------------------|---------|-------------------|------------------------------------------------------------------------------------------------------|
| `mask_folder`      | `str`   | *required*        | Path to the directory containing segmentation masks in `.nii.gz` format.                            |
| `output_path`      | `str`   | *required*        | Directory where bounding box masks and optional statistics will be saved.                           |
| `voxel_size`       | `tuple` | `(3.0, 3.0, 3.0)` | Physical voxel dimensions in millimeters as `(x, y, z)` for volume calculations.                   |
| `volume_threshold` | `float` | `1000.0`          | Minimum component volume in mm³ required to generate a bounding box.                                |
| `mask_value`       | `int`   | `1`               | Integer value in the mask representing the target region to analyze.                                |
| `save_stats`       | `bool`  | `True`            | If `True`, saves a CSV file with bounding box counts per file.                                      |
| `debug`            | `bool`  | `False`           | If `True`, prints detailed processing information and summary statistics.                           |

## Returns

`None` – The function saves NIfTI files and optional statistics to disk.

## Output Files

### Bounding Box Masks
Each input mask generates an output file:
```
<PREFIX>_bounding_boxes.nii.gz
```
where `<PREFIX>` is the original filename without the `.nii.gz` extension.

**Example**: Input `patient_042_mask.nii.gz` → Output `patient_042_mask_bounding_boxes.nii.gz`

### Statistics File (Optional)
When `save_stats=True`, a CSV file `bounding_boxes_stats.csv` is created with:

| Column                  | Description                                      |
|-------------------------|--------------------------------------------------|
| `FILENAME`              | Name of the input mask file                      |
| `NUM_BOUNDING_BOXES`    | Number of bounding boxes extracted from the file |
| `TOTAL_BOUNDING_BOXES`  | Sum across all files (last row)                  |

## Volume Filtering

The function filters connected components based on their physical volume:

**Volume Calculation**:
```
volume (mm³) = number_of_voxels × voxel_x × voxel_y × voxel_z
```

Only components with `volume ≥ volume_threshold` are included in the output. This helps:
- Remove noise and imaging artifacts
- Exclude small false positives
- Focus on clinically relevant structures
- Reduce processing time for downstream tasks

## Connected Component Detection

The function identifies each disconnected region with the specified `mask_value` as a separate component. This means:

- **Multiple instances**: Separate anatomical structures or lesions each get their own bounding box
- **Single region**: A fully connected structure generates one bounding box
- **Empty masks**: Files with no voxels matching `mask_value` produce empty output

## Exceptions

| Exception            | Condition                                                          |
|----------------------|--------------------------------------------------------------------|
| `FileNotFoundError`  | The `mask_folder` does not exist or contains no `.nii.gz` files   |

## Usage Notes

- **Input Format**: Only `.nii.gz` files are processed
- **3D Volumes Required**: Input masks must be 3D NIfTI images
- **Progress Display**: Shows progress bar during processing
- **Error Handling**: Files that fail to process are skipped with error messages
- **Output Directory**: Automatically created if it doesn't exist
- **Voxel Size Accuracy**: Ensure `voxel_size` matches your scan resolution for accurate volume filtering

## Examples

### Basic Usage
Extract bounding boxes with default settings:

```python
from nidataset.volume import extract_bounding_boxes_dataset

extract_bounding_boxes_dataset(
    mask_folder="dataset/segmentations/",
    output_path="dataset/bounding_boxes/",
    voxel_size=(3.0, 3.0, 3.0),
    volume_threshold=1000.0,
    mask_value=1,
    save_stats=True
)
# Creates: dataset/bounding_boxes/<mask>_bounding_boxes.nii.gz for each mask
# Creates: dataset/bounding_boxes/bounding_boxes_stats.csv
```

### High-Resolution Scans
Adjust voxel size for high-resolution imaging:

```python
extract_bounding_boxes_dataset(
    mask_folder="data/hr_masks/",
    output_path="data/hr_bboxes/",
    voxel_size=(0.5, 0.5, 1.0),  # High-resolution spacing
    volume_threshold=200.0,       # Lower threshold for smaller voxels
    mask_value=1,
    save_stats=True,
    debug=True
)
# Debug output shows total files and boxes extracted
```

### Multiple Mask Values
Process different anatomical structures separately:

```python
from nidataset.volume import extract_bounding_boxes_dataset

mask_folder = "dataset/multi_label_masks/"
output_base = "dataset/extracted_structures/"

# Dictionary of structures with their mask values and thresholds
structures = {
    'liver': {'value': 1, 'threshold': 50000.0},
    'kidney': {'value': 2, 'threshold': 15000.0},
    'spleen': {'value': 3, 'threshold': 8000.0}
}

for name, params in structures.items():
    print(f"Processing {name}...")
    extract_bounding_boxes_dataset(
        mask_folder=mask_folder,
        output_path=f"{output_base}/{name}/",
        voxel_size=(1.0, 1.0, 1.5),
        volume_threshold=params['threshold'],
        mask_value=params['value'],
        save_stats=True,
        debug=True
    )
```

### Lesion Detection with Aggressive Filtering
Filter out small false positives in lesion detection:

```python
extract_bounding_boxes_dataset(
    mask_folder="predictions/lesion_masks/",
    output_path="predictions/filtered_lesions/",
    voxel_size=(2.0, 2.0, 2.0),
    volume_threshold=5000.0,  # Aggressive filtering
    mask_value=1,
    save_stats=True,
    debug=True
)
print("Small false positives removed")
```

### Analyzing Statistics
Extract boxes and analyze the distribution:

```python
import pandas as pd
from nidataset.volume import extract_bounding_boxes_dataset

# Extract bounding boxes
extract_bounding_boxes_dataset(
    mask_folder="masks/",
    output_path="results/boxes/",
    voxel_size=(1.5, 1.5, 2.0),
    volume_threshold=1000.0,
    mask_value=1,
    save_stats=True
)

# Load and analyze statistics
stats = pd.read_csv("results/boxes/bounding_boxes_stats.csv")

# Remove total row for analysis
per_file = stats[stats['FILENAME'] != 'TOTAL_BOUNDING_BOXES'].copy()
per_file['NUM_BOUNDING_BOXES'] = pd.to_numeric(per_file['NUM_BOUNDING_BOXES'])

print("Bounding Box Statistics:")
print(f"  Total files: {len(per_file)}")
print(f"  Files with boxes: {(per_file['NUM_BOUNDING_BOXES'] > 0).sum()}")
print(f"  Average boxes per file: {per_file['NUM_BOUNDING_BOXES'].mean():.2f}")
print(f"  Max boxes in a file: {per_file['NUM_BOUNDING_BOXES'].max()}")

# Identify files without detections
no_boxes = per_file[per_file['NUM_BOUNDING_BOXES'] == 0]
if not no_boxes.empty:
    print(f"\nFiles with no bounding boxes ({len(no_boxes)}):")
    print(no_boxes['FILENAME'].tolist())
```

### Quality Control Workflow
Verify extraction results and identify issues:

```python
import pandas as pd
import nibabel as nib
from nidataset.volume import extract_bounding_boxes_dataset

# Extract bounding boxes
extract_bounding_boxes_dataset(
    mask_folder="qa/masks/",
    output_path="qa/boxes/",
    voxel_size=(1.0, 1.0, 1.0),
    volume_threshold=500.0,
    mask_value=1,
    save_stats=True,
    debug=True
)

# Load statistics
stats = pd.read_csv("qa/boxes/bounding_boxes_stats.csv")
per_file = stats[stats['FILENAME'] != 'TOTAL_BOUNDING_BOXES'].copy()
per_file['NUM_BOUNDING_BOXES'] = pd.to_numeric(per_file['NUM_BOUNDING_BOXES'])

# Check for outliers
mean_boxes = per_file['NUM_BOUNDING_BOXES'].mean()
std_boxes = per_file['NUM_BOUNDING_BOXES'].std()

outliers = per_file[
    per_file['NUM_BOUNDING_BOXES'] > mean_boxes + 2 * std_boxes
]

if not outliers.empty:
    print(f"Potential outliers detected ({len(outliers)} files):")
    for _, row in outliers.iterrows():
        print(f"  {row['FILENAME']}: {row['NUM_BOUNDING_BOXES']} boxes")
        
        # Verify by loading the output
        bbox_file = f"qa/boxes/{row['FILENAME'].replace('.nii.gz', '')}_bounding_boxes.nii.gz"
        bbox_img = nib.load(bbox_file)
        bbox_data = bbox_img.get_fdata()
        print(f"    Output shape: {bbox_data.shape}")
        print(f"    Non-zero voxels: {np.count_nonzero(bbox_data)}")
```

### Comparing Different Thresholds
Test multiple volume thresholds to find optimal filtering:

```python
from nidataset.volume import extract_bounding_boxes_dataset
import pandas as pd

mask_folder = "test_masks/"
thresholds = [500.0, 1000.0, 2000.0, 5000.0]

results = []

for threshold in thresholds:
    output_path = f"threshold_test/{int(threshold)}mm3/"
    
    extract_bounding_boxes_dataset(
        mask_folder=mask_folder,
        output_path=output_path,
        voxel_size=(1.0, 1.0, 1.0),
        volume_threshold=threshold,
        mask_value=1,
        save_stats=True
    )
    
    # Load statistics
    stats = pd.read_csv(f"{output_path}/bounding_boxes_stats.csv")
    total_row = stats[stats['FILENAME'] == 'TOTAL_BOUNDING_BOXES']
    total_boxes = int(total_row['NUM_BOUNDING_BOXES'].values[0])
    
    results.append({
        'threshold': threshold,
        'total_boxes': total_boxes
    })

# Display results
results_df = pd.DataFrame(results)
print("\nThreshold Comparison:")
print(results_df)
print("\nRecommendation: Choose threshold where boxes stabilize")
```

### Integration with Detection Pipeline
Use extracted bounding boxes for model evaluation:

```python
import pandas as pd
import nibabel as nib
from nidataset.volume import extract_bounding_boxes_dataset

# Extract ground truth bounding boxes
extract_bounding_boxes_dataset(
    mask_folder="ground_truth/masks/",
    output_path="ground_truth/boxes/",
    voxel_size=(1.0, 1.0, 1.0),
    volume_threshold=1000.0,
    mask_value=1,
    save_stats=True
)

# Load statistics
stats = pd.read_csv("ground_truth/boxes/bounding_boxes_stats.csv")
per_file = stats[stats['FILENAME'] != 'TOTAL_BOUNDING_BOXES'].copy()

print("Ground Truth Statistics:")
print(f"Total cases: {len(per_file)}")
print(f"Total lesions: {per_file['NUM_BOUNDING_BOXES'].sum()}")
print(f"Cases with lesions: {(per_file['NUM_BOUNDING_BOXES'] > 0).sum()}")
print(f"Average lesions per case: {per_file['NUM_BOUNDING_BOXES'].mean():.2f}")

# Use for evaluation
# - Compare with model predictions
# - Calculate detection metrics
# - Visualize bounding boxes in viewers
```

### Creating Training Data Subsets
Select cases based on bounding box counts:

```python
import pandas as pd
import shutil
from nidataset.volume import extract_bounding_boxes_dataset

# Extract bounding boxes
extract_bounding_boxes_dataset(
    mask_folder="all_masks/",
    output_path="all_boxes/",
    voxel_size=(1.5, 1.5, 2.0),
    volume_threshold=1500.0,
    mask_value=1,
    save_stats=True
)

# Load statistics
stats = pd.read_csv("all_boxes/bounding_boxes_stats.csv")
per_file = stats[stats['FILENAME'] != 'TOTAL_BOUNDING_BOXES'].copy()
per_file['NUM_BOUNDING_BOXES'] = pd.to_numeric(per_file['NUM_BOUNDING_BOXES'])

# Select cases with 1-5 lesions for training
training_cases = per_file[
    (per_file['NUM_BOUNDING_BOXES'] >= 1) & 
    (per_file['NUM_BOUNDING_BOXES'] <= 5)
]

print(f"Selected {len(training_cases)} cases for training")

# Copy selected cases to training folder
for filename in training_cases['FILENAME']:
    src = f"all_masks/{filename}"
    dst = f"training_data/masks/{filename}"
    shutil.copy(src, dst)
```

## Typical Workflow

```python
from nidataset.volume import extract_bounding_boxes_dataset
import pandas as pd

# 1. Define paths and parameters
mask_folder = "dataset/segmentation_masks/"
output_folder = "dataset/bounding_boxes/"
voxel_dims = (1.0, 1.0, 1.5)  # Your scan's voxel spacing
min_volume = 1000.0            # Minimum structure volume in mm³

# 2. Extract bounding boxes from all masks
extract_bounding_boxes_dataset(
    mask_folder=mask_folder,
    output_path=output_folder,
    voxel_size=voxel_dims,
    volume_threshold=min_volume,
    mask_value=1,
    save_stats=True,
    debug=True
)

# 3. Review statistics
stats = pd.read_csv(f"{output_folder}/bounding_boxes_stats.csv")
print("\nExtraction Summary:")
print(stats.tail())

# 4. Use bounding boxes for:
# - Visualization in medical imaging viewers
# - Validation of segmentation results
# - Region of interest extraction
# - Detection model evaluation
```