---
title: register_CTA
parent: Package Functions
nav_order: 9
---

# `register_CTA`

Register a CTA image to a template using mutual‑information optimisation. Produces a registered volume, a Gaussian‑filtered intermediate, and a transformation `.tfm`.

```python
register_CTA(
    nii_path: str,
    mask_path: str,
    template_path: str,
    template_mask_path: str,
    output_image_path: str,
    output_transformation_path: str,
    cleanup: bool = False,
    debug: bool = False
) -> None
```

#### Parameters

| Name                         | Type   | Description                                            |
| ---------------------------- | ------ | ------------------------------------------------------ |
| `nii_path`                   | `str`  | Input CTA volume.                                      |
| `mask_path`                  | `str`  | Corresponding brain mask.                              |
| `template_path`              | `str`  | Reference template image.                              |
| `template_mask_path`         | `str`  | Template mask.                                         |
| `output_image_path`          | `str`  | Folder for registered image and filtered intermediate. |
| `output_transformation_path` | `str`  | Folder for `.tfm` transform.                           |
| `cleanup`                    | `bool` | Delete Gaussian intermediate after success.            |
| `debug`                      | `bool` | Print saved paths when **True**.                       |

#### Returns

`None` – saves `_registered.nii.gz`, `_gaussian_filtered.nii.gz`, and `_transformation.tfm` under the specified output paths.

#### Example

```python
from nidataset.preprocessing import register_CTA

register_CTA(
    nii_path="cta_case01.nii.gz",
    mask_path="cta_case01_mask.nii.gz",
    template_path="template.nii.gz",
    template_mask_path="template_mask.nii.gz",
    output_image_path="results/registered/images/",
    output_transformation_path="results/registered/transforms/",
    cleanup=True,
    debug=False
)
```