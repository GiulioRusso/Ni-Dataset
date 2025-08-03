---
title: dataset_images_info
parent: Package Functions
nav_order: 15
---

# `dataset_images_info`

Collect metadata for every **NIfTI** file in a folder and save the summary as `dataset_images_info.csv`.

```python
dataset_images_info(
    nii_folder: str,
    output_path: str
) -> None
```

#### Parameters

| Name          | Type  | Description                                               |
| ------------- | ----- | --------------------------------------------------------- |
| `nii_folder`  | `str` | Directory containing input `.nii.gz` volumes.             |
| `output_path` | `str` | Folder where the CSV will be written (created if absent). |

#### Returns

`None` – writes a CSV with one row per file and columns:
`FILENAME, SHAPE (X, Y, Z), VOXEL SIZE (mm), DATA TYPE, MIN VALUE, MAX VALUE, BRAIN VOXELS, BRAIN VOLUME (mm³), BBOX MIN (X, Y, Z), BBOX MAX (X, Y, Z)`.

#### Example

```python
from nidataset.utility import dataset_images_info

dataset_images_info(
    nii_folder="dataset/",
    output_path="results/info/"
)
```