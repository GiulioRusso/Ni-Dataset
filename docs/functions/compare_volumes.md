---
title: compare_volumes
parent: Package Functions
nav_order: 33
---

# `compare_volumes`

Compute voxel-wise similarity metrics between two NIfTI volumes for evaluating segmentation quality, registration accuracy, or comparing annotations.

```python
compare_volumes(
    nii_path_a: str,
    nii_path_b: str,
    metrics: Optional[List[str]] = None,
    mask_path: Optional[str] = None,
    debug: bool = False
) -> Dict[str, float]
```

## Overview

This function computes a set of quantitative similarity and distance metrics between two NIfTI volumes. It supports both binary overlap metrics (Dice, Jaccard, Hausdorff) and intensity-based metrics (MSE, MAE, PSNR, correlation). Metrics can optionally be restricted to a masked region.

This is useful for:
- Evaluating segmentation accuracy against ground truth
- Measuring registration quality between aligned volumes
- Comparing different preprocessing pipelines
- Validating annotation consistency

## Supported Metrics

| Metric          | Type      | Range           | Description                                                         |
|-----------------|-----------|-----------------|---------------------------------------------------------------------|
| `dice`          | Binary    | [0, 1]          | Dice similarity coefficient (2×intersection / sum)                  |
| `jaccard`       | Binary    | [0, 1]          | Jaccard index / Intersection over Union                             |
| `hausdorff`     | Binary    | [0, ∞)          | Maximum surface distance in voxels                                  |
| `mse`           | Intensity | [0, ∞)          | Mean Squared Error                                                  |
| `mae`           | Intensity | [0, ∞)          | Mean Absolute Error                                                 |
| `psnr`          | Intensity | [0, ∞)          | Peak Signal-to-Noise Ratio (dB)                                     |
| `volume_diff`   | Binary    | [0, ∞)          | Absolute difference in non-zero voxel counts                        |
| `correlation`   | Intensity | [-1, 1]         | Pearson correlation coefficient                                     |

## Parameters

| Name         | Type            | Default    | Description                                                                                   |
|--------------|-----------------|------------|-----------------------------------------------------------------------------------------------|
| `nii_path_a` | `str`           | *required* | Path to the first `.nii.gz` file.                                                            |
| `nii_path_b` | `str`           | *required* | Path to the second `.nii.gz` file.                                                           |
| `metrics`    | `List[str]`     | `None`     | List of metric names to compute. If `None`, computes all available metrics.                  |
| `mask_path`  | `str`           | `None`     | Optional mask NIfTI file. If provided, metrics are computed only within the masked region.    |
| `debug`      | `bool`          | `False`    | If `True`, logs detailed metric results.                                                     |

## Returns

`Dict[str, float]` – Dictionary mapping metric names to their computed values.

## Exceptions

| Exception           | Condition                                                    |
|---------------------|--------------------------------------------------------------|
| `FileNotFoundError` | Any input file does not exist                                |
| `ValueError`        | Volumes have different shapes                                |
| `ValueError`        | Unknown metric names are requested                           |

## Usage Notes

- **Shape Requirement**: Both volumes must have identical shapes
- **Binary Metrics**: Dice, Jaccard, and Hausdorff treat any non-zero voxel as foreground
- **Hausdorff Distance**: Computed on full volumes even when a mask is provided
- **Perfect Match**: Dice and Jaccard return `1.0` when both volumes are empty
- **PSNR**: Returns `inf` when volumes are identical (MSE = 0)

## Examples

### Basic Usage
Compare a predicted segmentation against ground truth:

```python
from nidataset.analysis import compare_volumes

results = compare_volumes(
    nii_path_a="predictions/case001_pred.nii.gz",
    nii_path_b="ground_truth/case001_gt.nii.gz",
    metrics=["dice", "hausdorff"]
)
print(f"Dice: {results['dice']:.4f}")
print(f"Hausdorff: {results['hausdorff']:.2f} voxels")
```

### All Metrics
Compute every available metric:

```python
results = compare_volumes(
    nii_path_a="volume_a.nii.gz",
    nii_path_b="volume_b.nii.gz"
)
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")
```

### With Mask
Restrict comparison to a brain region:

```python
results = compare_volumes(
    nii_path_a="scan_a.nii.gz",
    nii_path_b="scan_b.nii.gz",
    mask_path="brain_mask.nii.gz",
    metrics=["mse", "psnr", "correlation"]
)
print(f"MSE within brain: {results['mse']:.4f}")
print(f"PSNR within brain: {results['psnr']:.2f} dB")
```

### Segmentation Evaluation
Evaluate multiple segmentation models:

```python
from nidataset.analysis import compare_volumes

models = ["unet", "vnet", "attention_unet"]
gt_path = "ground_truth/case001.nii.gz"

for model in models:
    pred_path = f"predictions/{model}/case001.nii.gz"
    results = compare_volumes(
        pred_path, gt_path,
        metrics=["dice", "jaccard", "hausdorff"]
    )
    print(f"{model}: Dice={results['dice']:.4f}, "
          f"Jaccard={results['jaccard']:.4f}, "
          f"Hausdorff={results['hausdorff']:.2f}")
```

## Typical Workflow

```python
from nidataset.analysis import compare_volumes

# 1. Define paths
prediction = "output/segmentation_pred.nii.gz"
ground_truth = "labels/segmentation_gt.nii.gz"

# 2. Choose metrics for your evaluation
results = compare_volumes(
    prediction, ground_truth,
    metrics=["dice", "jaccard", "hausdorff", "volume_diff"],
    debug=True
)

# 3. Analyze results
print(f"Overlap (Dice): {results['dice']:.4f}")
print(f"Overlap (Jaccard): {results['jaccard']:.4f}")
print(f"Max surface distance: {results['hausdorff']:.2f} voxels")
print(f"Volume difference: {results['volume_diff']} voxels")
```
