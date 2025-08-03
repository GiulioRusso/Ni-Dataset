---
title: skull_CTA
parent: Package Functions
nav_order: 21
---

### `generate_brain_mask_dataset`

Produce brain masks for every CTA scan in a directory.

```python
generate_brain_mask_dataset(
    nii_folder: str,
    output_path: str,
    threshold: tuple | None = None,
    closing_radius: int = 3,
    debug: bool = False
) -> None
```

#### Parameters

| Name             | Type            | Description                                |
| ---------------- | --------------- | ------------------------------------------ |
| `nii_folder`     | `str`           | Directory with CTA volumes.                |
| `output_path`    | `str`           | Destination for masks.                     |
| `threshold`      | `tuple` or `None` | Manual or automatic threshold (see above). |
| `closing_radius` | `int`           | Closing radius.                            |
| `debug`          | `bool`          | Verbose logging.                           |

#### Returns

`None` – saves one mask per input file.

#### Example

```python
from nidataset.volume import generate_brain_mask_dataset

generate_brain_mask_dataset(
    nii_folder="dataset/cta/",
    output_path="results/masks/",
    threshold=(50, 300),
    closing_radius=4
)
```