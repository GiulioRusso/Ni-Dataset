---
title: skull_CTA
parent: Package Functions
nav_order: 7
---


### `resampling`

Resample a volume to a specified voxel grid and save as `<filename>_resampled.nii.gz`.

```python
resampling(
    nii_path: str,
    output_path: str,
    desired_volume: tuple,
    debug: bool = False
) -> None
```

#### Parameters

| Name             | Type    | Description                                                     |
| ---------------- | ------- | --------------------------------------------------------------- |
| `nii_path`       | `str`   | Input `.nii.gz` volume.                                         |
| `output_path`    | `str`   | Folder in which to save the resampled file (created if absent). |
| `desired_volume` | `tuple` | Target size `(X, Y, Z)` in voxels.                              |
| `debug`          | `bool`  | Print the output path when **True**.                            |

#### Returns

`None` – writes the resampled image to `output_path`.

#### Example

```python
from nidataset.preprocessing import resampling

resampling(
    nii_path="cta_case01.nii.gz",
    output_path="results/resampled/",
    desired_volume=(224, 224, 128),
    debug=True
)
```