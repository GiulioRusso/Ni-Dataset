---
title: from_2D_to_3D_coords
parent: Package Functions
nav_order: 2
---

# `from_2D_to_3D_coords`

Convert 2‑D slice‑based coordinates to canonical **(X, Y, Z)** order for axial, coronal, or sagittal views.

```python
from_2D_to_3D_coords(
    df: pd.DataFrame,
    view: str
) -> pd.DataFrame
```

#### Parameters

| Name   | Type           | Description                                                                                                                                                           |
| ------ | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `df`   | `pd.DataFrame` | Accepted layouts:<br>• **3‑column** – `['X', 'Y', 'SLICE NUMBER']`<br>• **6‑column** – `['X MIN', 'Y MIN', 'SLICE NUMBER MIN', 'X MAX', 'Y MAX', 'SLICE NUMBER MAX']` |
| `view` | `str`          | One of `'axial'`, `'coronal'`, `'sagittal'`.                                                                                                                          |

#### Returns

`pd.DataFrame` – Copy of `df` with columns renamed to `['X', 'Y', 'Z']` **or** `['X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX']`, reordered so that **Z** is the slice index.

#### Example

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

axial_boxes = pd.DataFrame({
    "X MIN": [10], "Y MIN": [15], "SLICE NUMBER MIN": [5],
    "X MAX": [20], "Y MAX": [25], "SLICE NUMBER MAX": [10]
})

boxes_3d = from_2D_to_3D_coords(axial_boxes, view="axial")

print(boxes_3d.head())
```
