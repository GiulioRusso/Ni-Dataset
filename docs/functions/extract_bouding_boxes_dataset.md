---
title: skull_CTA
parent: Package Functions
nav_order: 19
---

### `extract_bounding_boxes_dataset`

Run `extract_bounding_boxes` on every mask in a folder and optionally record counts in `bounding_boxes_stats.csv`.

```python
extract_bounding_boxes_dataset(
    mask_folder: str,
    output_path: str,
    voxel_size: tuple = (3.0, 3.0, 3.0),
    volume_threshold: float = 1000.0,
    mask_value: int = 1,
    save_stats: bool = True,
    debug: bool = False
) -> None
```

#### Parameters

| Name               | Type    | Description                                        |
| ------------------ | ------- | -------------------------------------------------- |
| `mask_folder`      | `str`   | Directory containing `.nii.gz` masks.              |
| `output_path`      | `str`   | Destination for bounding‑box masks and stats file. |
| `voxel_size`       | `tuple` | Voxel spacing in mm.                               |
| `volume_threshold` | `float` | Component size cutoff.                             |
| `mask_value`       | `int`   | Label to analyse.                                  |
| `save_stats`       | `bool`  | Write per‑file counts when **True**.               |
| `debug`            | `bool`  | Verbose output.                                    |

#### Returns

`None` – writes masks and optional stats CSV.

#### Example

```python
from nidataset.volume import extract_bounding_boxes_dataset

extract_bounding_boxes_dataset(
    mask_folder="dataset/masks/",
    output_path="results/bboxes/",
    voxel_size=(1,1,1),
    volume_threshold=200.0,
    save_stats=True
)
```