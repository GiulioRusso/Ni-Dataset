---
title: draw_boxes
parent: 📦 Package Functions
nav_order: 1
---

### `draw_boxes`


Draw 3‑D bounding boxes on a reference **NIfTI** volume and save the result as `<filename>_boxes.nii.gz`.

```python
draw_boxes(
    df: pd.DataFrame,
    nii_path: str,
    output_path: str,
    intensity_based_on_score: bool = False,
    debug: bool = False
) -> None
```

#### Parameters

| Name                       | Type           | Description                                                                                                                                                     |
| -------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `df`                       | `pd.DataFrame` | Bounding‑box coordinates. Must contain:<br>• `'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'`<br>• Optional `'SCORE'` if `intensity_based_on_score=True`. |
| `nii_path`                 | `str`          | Path to the reference `.nii.gz` file used for shape and affine.                                                                                                 |
| `output_path`              | `str`          | Directory in which to write the new file (created automatically if missing).                                                                                    |
| `intensity_based_on_score` | `bool`         | **False** → every box voxel = `1`.<br>**True** → voxel intensity tiers: `≤0.50 → 1`, `≤0.75 → 2`, `>0.75 → 3`.                                                  |
| `debug`                    | `bool`         | If **True**, prints the final file path.                                                                                                                        |

#### Returns

`None` – writes the new NIfTI file to disk.

#### Example

```python
import pandas as pd
from nidataset.draw import draw_boxes

boxes = pd.DataFrame({
    "SCORE": [0.30, 0.80],
    "X MIN": [10, 40], "Y MIN": [12, 42], "Z MIN": [14, 44],
    "X MAX": [20, 50], "Y MAX": [22, 52], "Z MAX": [24, 54]
})

draw_boxes(
    df=boxes,
    nii_path="brain.nii.gz",
    output_path="results/",
    intensity_based_on_score=True
)
```
