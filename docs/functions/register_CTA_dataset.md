---
title: register_CTA_dataset
parent: Package Functions
nav_order: 10
---

# `register_CTA_dataset`

Batch‑register every CTA volume in a folder to a template, with optional per‑case output grouping.

```python
register_CTA_dataset(
    nii_folder: str,
    mask_folder: str,
    template_path: str,
    template_mask_path: str,
    output_image_path: str,
    output_transformation_path: str = "",
    saving_mode: str = "case",
    cleanup: bool = False,
    debug: bool = False
) -> None
```

#### Parameters

| Name                         | Type   | Description                                                                                         |
| ---------------------------- | ------ | --------------------------------------------------------------------------------------------------- |
| `nii_folder`                 | `str`  | Folder with input CTA volumes.                                                                      |
| `mask_folder`                | `str`  | Folder with corresponding brain masks.                                                              |
| `template_path`              | `str`  | Registration template image.                                                                        |
| `template_mask_path`         | `str`  | Template mask.                                                                                      |
| `output_image_path`          | `str`  | Destination for registered images.                                                                  |
| `output_transformation_path` | `str`  | Destination for `.tfm` files (ignored when `saving_mode='case'` as they are stored with the image). |
| `saving_mode`                | `str`  | `'case'` → sub‑folder per volume; `'folder'` → all results in shared dirs.                          |
| `cleanup`                    | `bool` | Remove Gaussian intermediates after each case.                                                      |
| `debug`                      | `bool` | Verbose progress output.                                                                            |

#### Returns

`None` – registers all volumes in `nii_folder`.

#### Example

```python
from nidataset.preprocessing import register_CTA_dataset

register_CTA_dataset(
    nii_folder="dataset/cta/",
    mask_folder="dataset/masks/",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_image_path="results/registered/images/",
    output_transformation_path="results/registered/transforms/",
    saving_mode="folder",
    cleanup=False,
    debug=False
)
```