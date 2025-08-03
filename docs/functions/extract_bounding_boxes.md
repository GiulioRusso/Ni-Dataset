---
title: extract_bounding_boxes
parent: Package Functions
nav_order: 18
---

# `extract_bounding_boxes`

Detect connected components in a segmentation mask, keep those above a volume threshold, and write a binary mask containing the 3‑D bounding boxes as `<filename>_bounding_boxes.nii.gz`.

```python
extract_bounding_boxes(
    mask_path: str,
    output_path: str,
    voxel_size: tuple = (3.0, 3.0, 3.0),
    volume_threshold: float = 1000.0,
    mask_value: int = 1,
    debug: bool = False
) -> None
```

#### Parameters

| Name               | Type    | Description                            |
| ------------------ | ------- | -------------------------------------- |
| `mask_path`        | `str`   | Input label map (`.nii.gz`).           |
| `output_path`      | `str`   | Folder for the bounding‑box mask.      |
| `voxel_size`       | `tuple` | Physical voxel size `(x, y, z)` mm.    |
| `volume_threshold` | `float` | Minimum component size in mm³ to keep. |
| `mask_value`       | `int`   | Label value to analyse.                |
| `debug`            | `bool`  | Print box count when **True**.         |

#### Returns

`None` – saves the bounding‑box mask to `output_path`.

#### Example

```python
from nidataset.volume import extract_bounding_boxes

extract_bounding_boxes(
    mask_path="lesion_mask.nii.gz",
    output_path="results/bboxes/",
    voxel_size=(0.5, 0.5, 0.5),
    volume_threshold=500.0
)
```