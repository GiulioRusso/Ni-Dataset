---
title: extract_slices
parent: Package Functions
nav_order: 11
---

# `extract_slices`

Extract all slices from a 3‑D **NIfTI** volume and save them as `.tif` images using the pattern

```
<filename>_<view>_<slice‑idx>.tif
```

```python
extract_slices(
    nii_path: str,
    output_path: str,
    view: str = "axial",
    debug: bool = False
) -> None
```

#### Parameters

| Name          | Type   | Description                                                      |
| ------------- | ------ | ---------------------------------------------------------------- |
| `nii_path`    | `str`  | Input `.nii.gz` volume.                                          |
| `output_path` | `str`  | Folder for the extracted images (created if absent).             |
| `view`        | `str`  | `'axial'` → Z‑axis, `'coronal'` → Y‑axis, `'sagittal'` → X‑axis. |
| `debug`       | `bool` | Print total number of slices when **True**.                      |

#### Returns

`None` – writes one `.tif` per slice to `output_path`.

#### Example

```python
from nidataset.slices import extract_slices

extract_slices(
    nii_path="brain.nii.gz",
    output_path="results/slices/",
    view="sagittal",
    debug=True
)
```