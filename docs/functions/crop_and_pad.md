---
title: skull_CTA
parent: Package Functions
nav_order: 22
---

### `crop_and_pad`

Crop a volume to its non‑zero bounding box, then symmetrically pad or further crop it to a fixed shape while preserving affine alignment. Saves `<filename>_cropped_padded.nii.gz`.

```python
crop_and_pad(
    nii_path: str,
    output_path: str,
    target_shape: tuple = (128, 128, 128),
    debug: bool = False
) -> None
```

#### Parameters

| Name           | Type    | Description                           |
| -------------- | ------- | ------------------------------------- |
| `nii_path`     | `str`   | Input volume.                         |
| `output_path`  | `str`   | Destination folder.                   |
| `target_shape` | `tuple` | Desired `(X, Y, Z)` after processing. |
| `debug`        | `bool`  | Print shapes when **True**.           |

#### Returns

`None` – writes the processed volume to `output_path`.

#### Example

```python
from nidataset.volume import crop_and_pad

crop_and_pad(
    nii_path="cta_case01.nii.gz",
    output_path="results/cropped/",
    target_shape=(160, 160, 96)
)
```