---
title: mip
parent: Package Functions
nav_order: 5
---

# `mip`

Generate a sliding‑window 3‑D Maximum Intensity Projection (MIP) from a single volume and save it as `<filename>_mip_<view>.nii.gz`.

```python
mip(
    nii_path: str,
    output_path: str,
    window_size: int = 10,
    view: str = "axial",
    debug: bool = False
) -> None
```

#### Parameters

| Name          | Type   | Description                                                      |
| ------------- | ------ | ---------------------------------------------------------------- |
| `nii_path`    | `str`  | Input `.nii.gz` volume.                                          |
| `output_path` | `str`  | Destination folder (created if needed).                          |
| `window_size` | `int`  | Number of neighbouring slices merged for each projection.        |
| `view`        | `str`  | `'axial'` → Z‑axis, `'coronal'` → Y‑axis, `'sagittal'` → X‑axis. |
| `debug`       | `bool` | Print saved file path when **True**.                             |

#### Returns

`None` – writes the MIP NIfTI to `output_path`.

#### Example

```python
from nidataset.preprocessing import mip

mip(
    nii_path="cta_case01.nii.gz",
    output_path="results/mip/",
    window_size=20,
    view="coronal",
    debug=True
)
```