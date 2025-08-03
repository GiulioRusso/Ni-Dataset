---
title: skull_CTA
parent: Package Functions
nav_order: 3
---

# `skull_CTA`

Apply a CTA‑oriented skull‑stripping pipeline (threshold → smoothing → FSL BET → clipping) to a single **NIfTI** file. The result is written as `<filename>.skulled.clipped.nii.gz` plus a corresponding mask.

```python
skull_CTA(
    nii_path: str,
    output_path: str,
    f_value: float = 0.1,
    clip_value: tuple = (0, 200),
    cleanup: bool = False,
    debug: bool = False
) -> None
```

#### Parameters

| Name          | Type    | Description                                                                      |
| ------------- | ------- | -------------------------------------------------------------------------------- |
| `nii_path`    | `str`   | Path to the input `.nii.gz` CTA volume.                                          |
| `output_path` | `str`   | Folder for intermediate and final outputs (created if missing).                  |
| `f_value`     | `float` | Fractional intensity threshold for **BET**.                                      |
| `clip_value`  | `tuple` | Intensity range `(min, max)` used to clamp brain voxels after skull‑stripping.   |
| `cleanup`     | `bool`  | If **True**, deletes intermediate images, retaining only mask and clipped brain. |
| `debug`       | `bool`  | If **True**, prints file paths when done.                                        |

#### Returns

`None` – writes the stripped image (`.skulled.clipped.nii.gz`) and its mask to `output_path`.

#### Example

```python
from nidataset.preprocessing import skull_CTA

skull_CTA(
    nii_path="cta_case01.nii.gz",
    output_path="results/case01/",
    f_value=0.15,
    clip_value=(0, 150),
    cleanup=True,
    debug=True
)
```