---
title: dataset_annotations_info
parent: Package Functions
nav_order: 16
---

# `dataset_annotations_info`

Extract 3‑D bounding boxes of a specific label value from every annotation map in a folder and save them as `dataset_annotations_info.csv`.

```python
dataset_annotations_info(
    nii_folder: str,
    output_path: str,
    annotation_value: int = 1
) -> None
```

#### Parameters

| Name               | Type  | Description                                                    |
| ------------------ | ----- | -------------------------------------------------------------- |
| `nii_folder`       | `str` | Directory with annotation volumes (`.nii.gz`).                 |
| `output_path`      | `str` | Folder where the CSV will be stored (created if needed).       |
| `annotation_value` | `int` | Pixel value representing the region of interest (default `1`). |

#### Returns

`None` – writes a CSV listing each file and its list of bounding boxes (`[xmin, ymin, zmin, xmax, ymax, zmax]`).

#### Example

```python
from nidataset.utility import dataset_annotations_info

dataset_annotations_info(
    nii_folder="dataset/labels/",
    output_path="results/bboxes/",
    annotation_value=1
)
```