---
title: skull_CTA_dataset
parent: Package Functions
nav_order: 4
---

# `skull_CTA_dataset`

Run the same skull‑stripping pipeline over every `.nii.gz` file in a folder, with optional per‑case or shared output layout.

```python
skull_CTA_dataset(
    nii_folder: str,
    output_path: str,
    f_value: float = 0.1,
    clip_value: tuple = (0, 200),
    cleanup: bool = False,
    saving_mode: str = "case",
    debug: bool = False
) -> None
```

#### Parameters

| Name          | Type    | Description                                                             |
| ------------- | ------- | ----------------------------------------------------------------------- |
| `nii_folder`  | `str`   | Folder containing input `.nii.gz` files.                                |
| `output_path` | `str`   | Destination for processed outputs.                                      |
| `f_value`     | `float` | BET fractional threshold passed to each call.                           |
| `clip_value`  | `tuple` | Intensity range `(min, max)` for voxel clipping.                        |
| `cleanup`     | `bool`  | Remove intermediates after each case.                                   |
| `saving_mode` | `str`   | `'case'` → sub‑folder per file; `'folder'` → all results in one folder. |
| `debug`       | `bool`  | Verbose progress messages.                                              |

#### Returns

`None` – processes every file in `nii_folder` and saves outputs under `output_path`.

#### Example

```python
from nidataset.preprocessing import skull_CTA_dataset

skull_CTA_dataset(
    nii_folder="dataset/cta/",
    output_path="results/skull_strip/",
    f_value=0.1,
    clip_value=(0, 200),
    cleanup=False,
    saving_mode="case",
    debug=True
)
```