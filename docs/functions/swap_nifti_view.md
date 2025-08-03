---
title: skull_CTA
parent: Package Functions
nav_order: 17
---

### `swap_nifti_views`

Re‑orient a 3‑D **NIfTI** volume from one anatomical view to another by permuting axes, rotating 90 °, and updating the affine. The result is saved as `<filename>_swapped_<source>_to_<target>.nii.gz`.

```python
swap_nifti_views(
    nii_path: str,
    output_path: str,
    source_view: str,
    target_view: str,
    debug: bool = False
) -> None
```

#### Parameters

| Name          | Type   | Description                                  |
| ------------- | ------ | -------------------------------------------- |
| `nii_path`    | `str`  | Input `.nii.gz` file.                        |
| `output_path` | `str`  | Destination folder (auto‑created).           |
| `source_view` | `str`  | One of `'axial'`, `'coronal'`, `'sagittal'`. |
| `target_view` | `str`  | Desired orientation — same choices as above. |
| `debug`       | `bool` | Print shapes and path when **True**.         |

#### Returns

`None` – writes the re‑oriented volume to `output_path`.

#### Example

```python
from nidataset.volume import swap_nifti_views

swap_nifti_views(
    nii_path="scan_axial.nii.gz",
    output_path="results/oriented/",
    source_view="axial",
    target_view="sagittal",
    debug=True
)
```