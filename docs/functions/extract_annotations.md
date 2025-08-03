---
title: extract_annotations
parent: Package Functions
nav_order: 13
---

# `extract_annotations`

Export bounding‑box annotations from a 3‑D label map into CSV files. Supports per‑slice or per‑volume output and either box centers or full coordinates.

```python
extract_annotations(
    nii_path: str,
    output_path: str,
    view: str = "axial",
    saving_mode: str = "slice",
    data_mode: str = "center",
    debug: bool = False
) -> None
```

#### Parameters

| Name          | Type   | Description                                                                                   |
| ------------- | ------ | --------------------------------------------------------------------------------------------- |
| `nii_path`    | `str`  | Input annotation volume (`.nii.gz`).                                                          |
| `output_path` | `str`  | Folder for CSV outputs (created if needed).                                                   |
| `view`        | `str`  | Alignment for 2‑D extraction: `'axial'`, `'coronal'`, `'sagittal'`.                           |
| `saving_mode` | `str`  | `'slice'` → one CSV per slice; `'volume'` → single CSV for entire scan.                       |
| `data_mode`   | `str`  | `'center'` → save `(x, y, z)` centers; `'box'` → save `(xmin, ymin, zmin, xmax, ymax, zmax)`. |
| `debug`       | `bool` | Print counts when **True**.                                                                   |

#### Returns

`None` – writes CSV files named either

```
<filename>_<view>_<slice‑idx>.csv  # when saving_mode='slice'
<filename>.csv                    # when saving_mode='volume'
```

#### Example

```python
from nidataset.slices import extract_annotations

extract_annotations(
    nii_path="brain_labels.nii.gz",
    output_path="results/annots/",
    view="sagittal",
    saving_mode="volume",
    data_mode="box",
    debug=True
)
```