---
title: extract_annotations_dataset
parent: Package Functions
nav_order: 14
---

# `extract_annotations_dataset`

Run `extract_annotations` over an entire dataset folder with optional per‑case or shared output, per‑slice or per‑volume extraction, and optional annotation statistics.

```python
extract_annotations_dataset(
    nii_folder: str,
    output_path: str,
    view: str = "axial",
    saving_mode: str = "case",
    extraction_mode: str = "slice",
    data_mode: str = "center",
    save_stats: bool = False
) -> None
```

#### Parameters

| Name              | Type   | Description                                                                        |
| ----------------- | ------ | ---------------------------------------------------------------------------------- |
| `nii_folder`      | `str`  | Directory with annotation volumes.                                                 |
| `output_path`     | `str`  | Destination root.                                                                  |
| `view`            | `str`  | `'axial'`, `'coronal'`, or `'sagittal'`.                                           |
| `saving_mode`     | `str`  | `'case'` → sub‑folder per file; `'view'` → one folder per view.                    |
| `extraction_mode` | `str`  | `'slice'` → CSV per slice; `'volume'` → single CSV per case.                       |
| `data_mode`       | `str`  | `'center'` or `'box'` (see above).                                                 |
| `save_stats`      | `bool` | If **True**, writes `<view>_annotations_stats.csv` with per‑case counts and total. |

#### Returns

`None` – extracts annotations for all cases and writes optional stats file.

#### Example

```python
from nidataset.slices import extract_annotations_dataset

extract_annotations_dataset(
    nii_folder="dataset/labels/",
    output_path="results/annots/",
    view="coronal",
    saving_mode="view",
    extraction_mode="volume",
    data_mode="center",
    save_stats=True
)
```