---
title: skull_CTA
parent: Package Functions
nav_order: 12
---

### `extract_slices_dataset`

Batch‑extract slices from every `.nii.gz` file in a folder, with optional per‑case or shared output layout and optional per‑case slice statistics.

```python
extract_slices_dataset(
    nii_folder: str,
    output_path: str,
    view: str = "axial",
    saving_mode: str = "case",
    save_stats: bool = False
) -> None
```

#### Parameters

| Name          | Type   | Description                                                                         |
| ------------- | ------ | ----------------------------------------------------------------------------------- |
| `nii_folder`  | `str`  | Folder with input volumes.                                                          |
| `output_path` | `str`  | Destination root for extracted images.                                              |
| `view`        | `str`  | `'axial'`, `'coronal'`, or `'sagittal'`.                                            |
| `saving_mode` | `str`  | `'case'` → sub‑folder per file; `'view'` → one folder per view.                     |
| `save_stats`  | `bool` | If **True**, writes `<view>_slices_stats.csv` containing counts per case and total. |

#### Returns

`None` – saves all slices and optional statistics file under `output_path`.

#### Example

```python
from nidataset.slices import extract_slices_dataset

extract_slices_dataset(
    nii_folder="dataset/",
    output_path="results/slices/",
    view="coronal",
    saving_mode="view",
    save_stats=True
)
```