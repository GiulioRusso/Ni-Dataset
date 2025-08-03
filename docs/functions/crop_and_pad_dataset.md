---
title: skull_CTA
parent: Package Functions
nav_order: 23
---

### `crop_and_pad_dataset`

Apply `crop_and_pad` to every file in a folder and optionally record original/final shapes in `crop_pad_stats.csv`.

```python
crop_and_pad_dataset(
    nii_folder: str,
    output_path: str,
    target_shape: tuple = (128, 128, 128),
    save_stats: bool = False
) -> None
```

#### Parameters

| Name           | Type    | Description                                |
| -------------- | ------- | ------------------------------------------ |
| `nii_folder`   | `str`   | Directory with CTA volumes.                |
| `output_path`  | `str`   | Destination for processed files and stats. |
| `target_shape` | `tuple` | Desired shape after processing.            |
| `save_stats`   | `bool`  | Write per‑file shape log when **True**.    |

#### Returns

`None` – processes all scans and optional stats CSV.

#### Example

```python
from nidataset.volume import crop_and_pad_dataset

crop_and_pad_dataset(
    nii_folder="dataset/cta/",
    output_path="results/cropped/",
    target_shape=(128,128,128),
    save_stats=True
)
```