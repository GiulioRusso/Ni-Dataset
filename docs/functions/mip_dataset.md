---
title: mip_dataset
parent: Package Functions
nav_order: 6
---


# `mip_dataset`

Create MIP volumes for every NIfTI file in a folder, optionally grouping outputs by case or by view.

```python
mip_dataset(
    nii_folder: str,
    output_path: str,
    window_size: int = 10,
    view: str = "axial",
    saving_mode: str = "case",
    debug: bool = False
) -> None
```

#### Parameters

| Name          | Type   | Description                                                     |
| ------------- | ------ | --------------------------------------------------------------- |
| `nii_folder`  | `str`  | Directory with `.nii.gz` files.                                 |
| `output_path` | `str`  | Destination root for MIP outputs.                               |
| `window_size` | `int`  | Window width used for each MIP.                                 |
| `view`        | `str`  | `'axial'`, `'coronal'`, or `'sagittal'`.                        |
| `saving_mode` | `str`  | `'case'` → sub‑folder per file; `'view'` → one folder per view. |
| `debug`       | `bool` | Verbose logging.                                                |

#### Returns

`None` – generates MIPs for all cases in `nii_folder`.

#### Example

```python
from nidataset.preprocessing import mip_dataset

mip_dataset(
    nii_folder="dataset/cta/",
    output_path="results/mip/",
    window_size=15,
    view="axial",
    saving_mode="view",
    debug=False
)
```