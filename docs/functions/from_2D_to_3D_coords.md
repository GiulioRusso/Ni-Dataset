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
| `df`   | `pd.DataFrame` | Accepted layouts:<br>• **3‑column** – `['X', 'Y', 'SLICE NUMBER']`<br>• **6‑column** – `['X_MIN', 'Y_MIN', 'SLICE_NUMBER_MIN', 'X_MAX', 'Y_MAX', 'SLICE_NUMBER_MAX']` |
| `view` | `str`          | One of `'axial'`, `'coronal'`, `'sagittal'`.                                                                                                                          |

#### Returns

`pd.DataFrame` – Copy of `df` with columns renamed to `['X', 'Y', 'Z']` **or** `['X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX']`, reordered so that **Z** is the slice index.

#### Example

```python
import pandas as pd
from nidataset.draw import from_2D_to_3D_coords

axial_boxes = pd.DataFrame({
    "X_MIN": [10], "Y_MIN": [15], "SLICE_NUMBER_MIN": [5],
    "X_MAX": [20], "Y_MAX": [25], "SLICE_NUMBER_MAX": [10]
})

boxes_3d = from_2D_to_3D_coords(axial_boxes, view="axial")

print(boxes_3d.head())
```
