---
title: from_2D_to_3D_coords
parent: Package Functions
nav_order: 13
---

# `from_2D_to_3D_coords`

Convert 2D slice-based coordinates to 3D volumetric coordinates based on anatomical view.

```python
from_2D_to_3D_coords(
    df: pd.DataFrame,
    view: str
) -> pd.DataFrame
```

## Overview

This function transforms 2D coordinates extracted from a specific anatomical view (axial, coronal, or sagittal) into a unified 3D coordinate system using the (X, Y, Z) axis convention. It handles both point coordinates and bounding box coordinates, performing the necessary axis reordering to ensure consistency across different anatomical planes.

**Coordinate transformation pipeline**:
1. **Validation**: Checks dataframe format, column names, and view specification
2. **Axis reordering**: Swaps coordinate axes based on anatomical view
3. **Column renaming**: Standardizes to X, Y, Z convention
4. **Output**: Returns transformed dataframe with proper 3D coordinates

This is essential for:
- Converting slice-based annotations to volumetric coordinates
- Standardizing coordinates across different anatomical views
- Preparing 2D detection results for 3D visualization
- Integrating slice-wise predictions into 3D bounding boxes
- Working with `draw_3D_boxes` and other 3D processing functions

## Parameters

| Name   | Type           | Default    | Description                                                                                          |
|--------|----------------|------------|------------------------------------------------------------------------------------------------------|
| `df`   | `pd.DataFrame` | *required* | DataFrame with 2D slice-based coordinates. Must be 3-column or 6-column format.                     |
| `view` | `str`          | *required* | Anatomical view: `'axial'`, `'coronal'`, or `'sagittal'`.                                           |

## Input DataFrame Formats

The function accepts two coordinate formats:

### 3-Column Format (Point Coordinates)

Single point per row with slice number:

| Column          | Description                           |
|-----------------|---------------------------------------|
| `X`             | X coordinate in slice plane           |
| `Y`             | Y coordinate in slice plane           |
| `SLICE_NUMBER`  | Slice index                           |

**Example**:
```python
df = pd.DataFrame({
    'X': [45, 67],
    'Y': [32, 89],
    'SLICE_NUMBER': [12, 15]
})
```

### 6-Column Format (Bounding Box Coordinates)

Bounding box per row with slice range:

| Column               | Description                           |
|----------------------|---------------------------------------|
| `X_MIN`              | Minimum X coordinate in slice plane   |
| `Y_MIN`              | Minimum Y coordinate in slice plane   |
| `SLICE_NUMBER_MIN`   | Starting slice index                  |
| `X_MAX`              | Maximum X coordinate in slice plane   |
| `Y_MAX`              | Maximum Y coordinate in slice plane   |
| `SLICE_NUMBER_MAX`   | Ending slice index                    |

**Example**:
```python
df = pd.DataFrame({
    'X_MIN': [10, 30],
    'Y_MIN': [15, 35],
    'SLICE_NUMBER_MIN': [5, 10],
    'X_MAX': [20, 40],
    'Y_MAX': [25, 45],
    'SLICE_NUMBER_MAX': [8, 12]
})
```

## Returns

`pd.DataFrame` — Copy of input dataframe with columns renamed and reordered according to the anatomical view.

### Output Formats

**For 3-column input**:
- Columns: `['X', 'Y', 'Z']`
- Z represents the through-plane axis

**For 6-column input**:
- Columns: `['X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX']`
- Z_MIN and Z_MAX represent the through-plane extent

## Anatomical View Mappings

### Axial View (Transverse Plane)

**3-Column Format**:
```
Input:  X, Y, SLICE_NUMBER
         ↓  ↓       ↓
Output: Y, X, SLICE_NUMBER
Rename: X, Y,       Z
```

**6-Column Format**:
```
Input:  X_MIN, Y_MIN, SLICE_NUMBER_MIN, X_MAX, Y_MAX, SLICE_NUMBER_MAX
          ↓      ↓            ↓            ↓      ↓            ↓
Output: Y_MIN, X_MIN, SLICE_NUMBER_MIN, Y_MAX, X_MAX, SLICE_NUMBER_MAX
Rename: X_MIN, Y_MIN,       Z_MIN,      X_MAX, Y_MAX,       Z_MAX
```

**Transformation**: Swaps X and Y coordinates

### Coronal View (Frontal Plane)

**3-Column Format**:
```
Input:        X,         Y, SLICE_NUMBER
              ↓          ↓       ↓
Output: SLICE_NUMBER,    X,      Y
Rename:       X,         Y,      Z
```

**6-Column Format**:
```
Input:  SLICE_NUMBER_MIN, X_MIN, Y_MIN, SLICE_NUMBER_MAX, X_MAX, Y_MAX
               ↓             ↓      ↓            ↓            ↓      ↓
Output:      X_MIN,       Y_MIN, Z_MIN,       X_MAX,      Y_MAX, Z_MAX
```

**Transformation**: SLICE_NUMBER → X, X → Y, Y → Z

### Sagittal View (Lateral Plane)

**3-Column Format**:
```
Input:        X, SLICE_NUMBER,   Y
              ↓       ↓          ↓
Output: SLICE_NUMBER, X,         Y
Rename:       X,      Y,         Z
```

**6-Column Format**:
```
Input:  SLICE_NUMBER_MIN, X_MIN, Y_MIN, SLICE_NUMBER_MAX, X_MAX, Y_MAX
               ↓             ↓      ↓            ↓            ↓      ↓
Output:      X_MIN,       Y_MIN, Z_MIN,       X_MAX,      Y_MAX, Z_MAX
```

**Transformation**: SLICE_NUMBER → X, X → Y, Y → Z

## Important Notes

### Column Name Requirements
- Input columns must exactly match expected names (case-sensitive)
- `SLICE_NUMBER` must be used (not `SLICE`, `SLICE_INDEX`, etc.)
- For 6-column format, all coordinates must have `_MIN` and `_MAX` suffixes
- Column order in input does not matter

### Coordinate System
- All coordinates are in voxel indices (integer values)
- Coordinates are 0-indexed
- Output coordinates ready for use with `draw_3D_boxes`
- Axis reordering depends on view (see mappings above)

### View Specification
- View names are case-sensitive and must be lowercase
- Only accepts: `'axial'`, `'coronal'`, `'sagittal'`
- Invalid view names raise `ValueError`

## Exceptions

| Exception     | Condition                                                          |
|---------------|--------------------------------------------------------------------|
| `ValueError`  | Invalid view name (not axial, coronal, or sagittal)               |
| `ValueError`  | DataFrame has wrong number of columns (not 3 or 6)                |
| `ValueError`  | Required columns missing from input dataframe                      |

## Usage Notes

- **Immutable Input**: Original dataframe is not modified (returns a copy)
- **Column Order**: Output follows standardized X, Y, Z ordering
- **View Consistency**: Ensure view matches how slices were originally annotated
- **Integration**: Output format directly compatible with `draw_3D_boxes`
- **Batch Processing**: Can process multiple annotations in single dataframe

## Examples

### Basic Usage - Axial View (3-column)
Convert axial slice coordinates to 3D:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Axial view point coordinates
axial_points = pd.DataFrame({
    'X': [45, 67, 89],
    'Y': [32, 54, 76],
    'SLICE_NUMBER': [10, 15, 20]
})

points_3d = from_2D_to_3D_coords(axial_points, view='axial')
print(points_3d)
#    X   Y   Z
# 0  32  45  10  (Note: X and Y are swapped)
# 1  54  67  15
# 2  76  89  20
```

### Basic Usage - Axial View (6-column)
Convert axial bounding boxes to 3D:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Axial bounding boxes
axial_boxes = pd.DataFrame({
    'X_MIN': [10, 30],
    'Y_MIN': [15, 35],
    'SLICE_NUMBER_MIN': [5, 10],
    'X_MAX': [20, 40],
    'Y_MAX': [25, 45],
    'SLICE_NUMBER_MAX': [8, 15]
})

boxes_3d = from_2D_to_3D_coords(axial_boxes, view='axial')
print(boxes_3d)
#    X_MIN  Y_MIN  Z_MIN  X_MAX  Y_MAX  Z_MAX
# 0     15     10      5     25     20      8  (X and Y swapped)
# 1     35     30     10     45     40     15
```

### Coronal View Conversion
Handle coronal (frontal plane) annotations:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Coronal view bounding boxes
coronal_boxes = pd.DataFrame({
    'X_MIN': [20, 50],
    'Y_MIN': [10, 30],
    'SLICE_NUMBER_MIN': [15, 25],
    'X_MAX': [30, 60],
    'Y_MAX': [20, 40],
    'SLICE_NUMBER_MAX': [20, 30]
})

boxes_3d = from_2D_to_3D_coords(coronal_boxes, view='coronal')
print(boxes_3d)
#    X_MIN  Y_MIN  Z_MIN  X_MAX  Y_MAX  Z_MAX
# 0     15     20     10     20     30     20  (SLICE→X, X→Y, Y→Z)
# 1     25     50     30     30     60     40
```

### Sagittal View Conversion
Handle sagittal (lateral plane) annotations:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Sagittal view bounding boxes
sagittal_boxes = pd.DataFrame({
    'X_MIN': [30, 60],
    'Y_MIN': [40, 70],
    'SLICE_NUMBER_MIN': [10, 20],
    'X_MAX': [40, 70],
    'Y_MAX': [50, 80],
    'SLICE_NUMBER_MAX': [15, 25]
})

boxes_3d = from_2D_to_3D_coords(sagittal_boxes, view='sagittal')
print(boxes_3d)
#    X_MIN  Y_MIN  Z_MIN  X_MAX  Y_MAX  Z_MAX
# 0     10     30     40     15     40     50  (SLICE→X, X→Y, Y→Z)
# 1     20     60     70     25     70     80
```

### Integration with draw_3D_boxes
Complete workflow from 2D annotations to 3D visualization:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords, draw_3D_boxes

# Step 1: Load 2D annotations (axial slices)
annotations_2d = pd.DataFrame({
    'SCORE': [0.85, 0.92, 0.78],
    'X_MIN': [45, 67, 89],
    'Y_MIN': [32, 54, 76],
    'SLICE_NUMBER_MIN': [10, 15, 20],
    'X_MAX': [55, 77, 99],
    'Y_MAX': [42, 64, 86],
    'SLICE_NUMBER_MAX': [12, 17, 22]
})

# Step 2: Convert to 3D coordinates
# Note: SCORE column is preserved during conversion
annotations_3d = from_2D_to_3D_coords(annotations_2d, view='axial')

# Step 3: Draw boxes on volume
draw_3D_boxes(
    df=annotations_3d,
    nii_path='scan.nii.gz',
    output_path='output/',
    intensity_based_on_score=True,
    debug=True
)
```

### Process Multiple Views
Convert annotations from different views:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Axial annotations
axial_df = pd.DataFrame({
    'X_MIN': [10], 'Y_MIN': [15], 'SLICE_NUMBER_MIN': [5],
    'X_MAX': [20], 'Y_MAX': [25], 'SLICE_NUMBER_MAX': [10]
})

# Coronal annotations
coronal_df = pd.DataFrame({
    'X_MIN': [30], 'Y_MIN': [35], 'SLICE_NUMBER_MIN': [12],
    'X_MAX': [40], 'Y_MAX': [45], 'SLICE_NUMBER_MAX': [17]
})

# Sagittal annotations
sagittal_df = pd.DataFrame({
    'X_MIN': [50], 'Y_MIN': [55], 'SLICE_NUMBER_MIN': [20],
    'X_MAX': [60], 'Y_MAX': [65], 'SLICE_NUMBER_MAX': [25]
})

# Convert all to 3D
axial_3d = from_2D_to_3D_coords(axial_df, view='axial')
coronal_3d = from_2D_to_3D_coords(coronal_df, view='coronal')
sagittal_3d = from_2D_to_3D_coords(sagittal_df, view='sagittal')

# Combine all annotations
all_boxes = pd.concat([axial_3d, coronal_3d, sagittal_3d], ignore_index=True)
print(f"Total boxes: {len(all_boxes)}")
```

### Preserve Additional Columns
Keep extra columns during conversion:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Annotations with additional metadata
annotations = pd.DataFrame({
    'CASE_ID': ['patient001', 'patient002'],
    'SCORE': [0.85, 0.92],
    'LABEL': ['lesion', 'tumor'],
    'X_MIN': [10, 30],
    'Y_MIN': [15, 35],
    'SLICE_NUMBER_MIN': [5, 10],
    'X_MAX': [20, 40],
    'Y_MAX': [25, 45],
    'SLICE_NUMBER_MAX': [8, 15]
})

# Convert to 3D (preserves CASE_ID, SCORE, LABEL)
annotations_3d = from_2D_to_3D_coords(annotations, view='axial')

print(annotations_3d.columns)
# Output: ['CASE_ID', 'SCORE', 'LABEL', 'X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX']
```

### Batch Processing Slice Annotations
Convert entire dataset of slice-based detections:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords
import os

def convert_slice_annotations(annotation_file, view, output_file):
    """Convert slice-based annotations to 3D coordinates."""
    
    # Load 2D annotations
    df_2d = pd.read_csv(annotation_file)
    
    # Convert to 3D
    df_3d = from_2D_to_3D_coords(df_2d, view=view)
    
    # Save 3D annotations
    df_3d.to_csv(output_file, index=False)
    
    print(f"Converted {len(df_3d)} annotations from {view} view")
    return df_3d

# Process multiple annotation files
annotation_files = [
    ('lesion_annotations_axial.csv', 'axial'),
    ('vessel_annotations_coronal.csv', 'coronal'),
    ('tumor_annotations_sagittal.csv', 'sagittal')
]

for ann_file, view in annotation_files:
    output_file = ann_file.replace('.csv', '_3d.csv')
    convert_slice_annotations(ann_file, view, output_file)
```

### Validate Conversion Results
Check that coordinates are properly transformed:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

def validate_conversion(df_2d, view):
    """Validate coordinate conversion."""
    
    df_3d = from_2D_to_3D_coords(df_2d, view=view)
    
    # Check column names
    if len([c for c in df_2d.columns if c in ['X', 'Y', 'SLICE_NUMBER']]) == 3:
        expected_cols = ['X', 'Y', 'Z']
    else:
        expected_cols = ['X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX']
    
    coord_cols = [c for c in df_3d.columns if c in expected_cols]
    assert coord_cols == expected_cols, f"Column names don't match expected format. Got {coord_cols}"
    
    # Check row count preserved
    assert len(df_3d) == len(df_2d), "Row count changed during conversion"
    
    # Check no NaN values introduced in coordinate columns
    assert not df_3d[expected_cols].isnull().any().any(), "NaN values introduced"
    
    print(f"Validation passed for {view} view")
    print(f"  Rows: {len(df_3d)}")
    print(f"  Coordinate columns: {expected_cols}")
    
    return True

# Test validation
test_df = pd.DataFrame({
    'X_MIN': [10], 'Y_MIN': [15], 'SLICE_NUMBER_MIN': [5],
    'X_MAX': [20], 'Y_MAX': [25], 'SLICE_NUMBER_MAX': [10]
})

validate_conversion(test_df, 'axial')
validate_conversion(test_df, 'coronal')
validate_conversion(test_df, 'sagittal')
```

### Error Handling
Handle invalid inputs gracefully:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

def safe_convert_coords(df, view):
    """Convert coordinates with error handling."""
    
    try:
        # Validate view
        valid_views = ['axial', 'coronal', 'sagittal']
        if view not in valid_views:
            print(f"Error: Invalid view '{view}'. Must be one of {valid_views}")
            return None
        
        # Check column count
        if df.shape[1] not in [3, 6]:
            # Count coordinate columns
            coord_cols = [c for c in df.columns 
                         if c in ['X', 'Y', 'SLICE_NUMBER', 
                                 'X_MIN', 'Y_MIN', 'SLICE_NUMBER_MIN',
                                 'X_MAX', 'Y_MAX', 'SLICE_NUMBER_MAX']]
            if len(coord_cols) not in [3, 6]:
                print(f"Error: Need 3 or 6 coordinate columns, found {len(coord_cols)}")
                return None
        
        # Convert
        df_3d = from_2D_to_3D_coords(df, view=view)
        print(f"Successfully converted {len(df_3d)} annotations")
        return df_3d
    
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        return None

# Test error handling
invalid_df = pd.DataFrame({'A': [1], 'B': [2]})
result = safe_convert_coords(invalid_df, 'axial')  # Returns None

valid_df = pd.DataFrame({
    'X': [10], 'Y': [15], 'SLICE_NUMBER': [5]
})
result = safe_convert_coords(valid_df, 'invalid_view')  # Returns None
result = safe_convert_coords(valid_df, 'axial')  # Works
```

### Working with Detection Model Output
Convert model predictions to 3D:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Simulated detection model output (per-slice predictions)
detections_axial = pd.DataFrame({
    'SCORE': [0.92, 0.87, 0.95, 0.78, 0.84],
    'X_MIN': [45, 67, 34, 89, 23],
    'Y_MIN': [32, 54, 28, 76, 45],
    'SLICE_NUMBER_MIN': [10, 10, 11, 15, 18],
    'X_MAX': [55, 77, 44, 99, 33],
    'Y_MAX': [42, 64, 38, 86, 55],
    'SLICE_NUMBER_MAX': [10, 10, 11, 15, 18]
})

# Convert to 3D
detections_3d = from_2D_to_3D_coords(detections_axial, view='axial')

# Filter by score threshold
high_conf_detections = detections_3d[detections_3d['SCORE'] > 0.85]

print(f"Total detections: {len(detections_3d)}")
print(f"High-confidence detections: {len(high_conf_detections)}")

# Visualize
from nidataset.draw import draw_3D_boxes

draw_3D_boxes(
    df=high_conf_detections,
    nii_path='scan.nii.gz',
    output_path='detections/',
    intensity_based_on_score=True
)
```

### Compare Coordinate Transformations
Understand how coordinates change across views:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Same 2D coordinates
original = pd.DataFrame({
    'X_MIN': [10], 'Y_MIN': [20], 'SLICE_NUMBER_MIN': [30],
    'X_MAX': [15], 'Y_MAX': [25], 'SLICE_NUMBER_MAX': [35]
})

print("Original 2D coordinates:")
print(original)
print()

print("Axial view transformation:")
axial = from_2D_to_3D_coords(original.copy(), view='axial')
print(axial)
print("  X and Y coordinates are swapped")
print()

print("Coronal view transformation:")
coronal = from_2D_to_3D_coords(original.copy(), view='coronal')
print(coronal)
print("  SLICE_NUMBER→X, X→Y, Y→Z")
print()

print("Sagittal view transformation:")
sagittal = from_2D_to_3D_coords(original.copy(), view='sagittal')
print(sagittal)
print("  SLICE_NUMBER→X, X→Y, Y→Z")
```

### Create Multi-View Visualization
Generate boxes from all three views:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords, draw_3D_boxes

# Annotations from different views
annotations = {
    'axial': pd.DataFrame({
        'SCORE': [0.9],
        'X_MIN': [40], 'Y_MIN': [50], 'SLICE_NUMBER_MIN': [60],
        'X_MAX': [50], 'Y_MAX': [60], 'SLICE_NUMBER_MAX': [70]
    }),
    'coronal': pd.DataFrame({
        'SCORE': [0.85],
        'X_MIN': [30], 'Y_MIN': [40], 'SLICE_NUMBER_MIN': [80],
        'X_MAX': [40], 'Y_MAX': [50], 'SLICE_NUMBER_MAX': [90]
    }),
    'sagittal': pd.DataFrame({
        'SCORE': [0.88],
        'X_MIN': [20], 'Y_MIN': [30], 'SLICE_NUMBER_MIN': [100],
        'X_MAX': [30], 'Y_MAX': [40], 'SLICE_NUMBER_MAX': [110]
    })
}

# Convert all views to 3D
all_boxes_3d = []
for view, df in annotations.items():
    df_3d = from_2D_to_3D_coords(df, view=view)
    all_boxes_3d.append(df_3d)
    print(f"{view.capitalize()} view: {len(df_3d)} boxes")

# Combine
combined = pd.concat(all_boxes_3d, ignore_index=True)

# Visualize all
draw_3D_boxes(
    df=combined,
    nii_path='scan.nii.gz',
    output_path='multi_view_boxes/',
    intensity_based_on_score=True
)

print(f"\nTotal boxes from all views: {len(combined)}")
```

### Point vs Bounding Box Conversion
Handle both formats:

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

# Point coordinates (3-column)
points_2d = pd.DataFrame({
    'X': [45, 67],
    'Y': [32, 54],
    'SLICE_NUMBER': [10, 15]
})

points_3d = from_2D_to_3D_coords(points_2d, view='axial')
print("Point coordinates:")
print(points_3d)
print()

# Bounding box coordinates (6-column)
boxes_2d = pd.DataFrame({
    'X_MIN': [45, 67],
    'Y_MIN': [32, 54],
    'SLICE_NUMBER_MIN': [10, 15],
    'X_MAX': [55, 77],
    'Y_MAX': [42, 64],
    'SLICE_NUMBER_MAX': [12, 17]
})

boxes_3d = from_2D_to_3D_coords(boxes_2d, view='axial')
print("Bounding box coordinates:")
print(boxes_3d)
```

## Typical Workflow

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords, draw_3D_boxes

# Step 1: Load 2D slice annotations
annotations_2d = pd.read_csv('detections_axial.csv')

# Step 2: Convert to 3D coordinates
annotations_3d = from_2D_to_3D_coords(annotations_2d, view='axial')

# Step 3: Visualize on 3D volume
draw_3D_boxes(
    df=annotations_3d,
    nii_path='scan.nii.gz',
    output_path='output/',
    intensity_based_on_score=True
)

# Step 4: Use 3D coordinates for analysis
# - Calculate 3D volumes
# - Perform 3D non-maximum suppression
# - Export for 3D viewers
```

## View Selection Guide

| Use Case | Recommended View | Reason |
|----------|------------------|--------|
| Brain CT/MRI (standard) | `axial` | Most common acquisition plane |
| Spine imaging | `sagittal` | Best shows vertebral column |
| Chest X-ray follow-up | `coronal` | Matches frontal X-ray view |
| Cardiac imaging | `axial` | Standard for cross-sectional heart views |
| Facial/sinus imaging | `coronal` | Shows paranasal sinuses well |
| Limb imaging | `sagittal` or `coronal` | Depends on limb orientation |

## Coordinate Transformation Summary

| View      | Input (2D)                    | Output (3D)                   | Transformation |
|-----------|-------------------------------|-------------------------------|----------------|
| Axial     | X, Y, SLICE_NUMBER            | Y, X, SLICE_NUMBER            | Swap X and Y   |
| Coronal   | X, Y, SLICE_NUMBER            | SLICE_NUMBER, X, Y            | Rotate axes    |
| Sagittal  | X, Y, SLICE_NUMBER            | SLICE_NUMBER, X, Y            | Rotate axes    |

After transformation, columns are renamed to X, Y, Z respectively.

## Troubleshooting

### Common Issues and Solutions

**Issue**: Column names not recognized
- **Solution**: Ensure exact column names match requirements
- Check for extra spaces or case differences
- Column names are case-sensitive
- Use `df.columns.tolist()` to verify names

**Issue**: ValueError about column count
- **Solution**: Must have exactly 3 or 6 coordinate columns
- Cannot mix point and bounding box formats
- Additional metadata columns are allowed

**Issue**: Coordinates seem incorrect after conversion
- **Solution**: Remember that axial view swaps X and Y
- Verify view parameter matches annotation source
- Check original imaging plane

**Issue**: Extra columns lost during conversion
- **Solution**: Extra columns are preserved automatically
- Only coordinate columns are transformed
- Check that column names don't conflict with coordinate names

**Issue**: Wrong view selected
- **Solution**: Determine view from slice acquisition:
  - Axial: Horizontal slices (top-down)
  - Coronal: Frontal slices (front-back)
  - Sagittal: Side slices (left-right)

## Performance Considerations

- **Memory**: Creates a copy of the dataframe
- **Speed**: Very fast, primarily involves column reordering
- **Batch Size**: Can handle thousands of annotations efficiently
- **Large Datasets**: Consider processing in chunks if memory constrained