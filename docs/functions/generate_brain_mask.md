---
title: skull_CTA
parent: Package Functions
nav_order: 20
---

### `generate_brain_mask`

Create a rough brain mask from a CTA scan using intensity thresholding, hole filling, closing, and largest‑component selection. Saves `<filename>_brain_mask.nii.gz`.

```python
generate_brain_mask(
    nii_path: str,
    output_path: str,
    threshold: tuple | None = None,
    closing_radius: int = 3,
    debug: bool = False
) -> None
```

#### Parameters

| Name             | Type            | Description                                                               |
| ---------------- | --------------- | ------------------------------------------------------------------------- |
| `nii_path`       | `str`           | Input CTA (`.nii.gz`).                                                    |
| `output_path`    | `str`           | Folder for the mask.                                                      |
| `threshold`      | `tuple` or `None` | `(low, high)` intensity range; if `None`, an adaptive Otsu range is used. |
| `closing_radius` | `int`           | Radius (voxels) for morphological closing.                                |
| `debug`          | `bool`          | Print threshold and path when **True**.                                   |

#### Returns

`None` – writes the mask to `output_path`.

#### Example

```python
from nidataset.volume import generate_brain_mask

generate_brain_mask(
    nii_path="cta_case01.nii.gz",
    output_path="results/masks/",
    threshold=None,
    closing_radius=5,
    debug=True
)
```