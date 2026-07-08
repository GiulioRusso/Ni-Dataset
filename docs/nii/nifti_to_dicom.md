---
title: nifti_to_dicom
parent: Quick Operations (nii)
nav_order: 11
---

# `nifti_to_dicom`

Convert a NIfTI volume into a DICOM series (one `.dcm` file per slice).

```python
nifti_to_dicom(
    nii_path: str,
    output_path: str,
    series_description: str = "nidataset",
    debug: bool = False
) -> str
```

## Overview

Writes a NIfTI volume as a DICOM series. Geometry (spacing, origin, orientation) is
preserved and float data is cast to `int16` (standard for CT/MR storage). Slices
share a generated `SeriesInstanceUID` and carry per-slice position and instance
tags, so viewers load them as one coherent volume.

A folder named after the input volume is created under `output_path`, containing
one `.dcm` per slice (`0000.dcm`, `0001.dcm`, …).

## Parameters

| Name                 | Type   | Default        | Description                                            |
|----------------------|--------|----------------|--------------------------------------------------------|
| `nii_path`           | `str`  | *required*     | Path to the input `.nii.gz` file.                     |
| `output_path`        | `str`  | *required*     | Directory under which the series folder is created.   |
| `series_description` | `str`  | `"nidataset"`  | DICOM SeriesDescription tag (0008,103E).              |
| `debug`              | `bool` | `False`        | If `True`, logs the output directory and slice count. |

## Returns

`str` – Path to the created DICOM series directory (`<output_path>/<PREFIX>/`).

## Output

```
<output_path>/<PREFIX>/
├── 0000.dcm
├── 0001.dcm
└── ...
```

## Exceptions

| Exception           | Condition                                    |
|---------------------|----------------------------------------------|
| `FileNotFoundError` | Input file does not exist                    |
| `ValueError`        | Input file is not a valid `.nii.gz` / `.nii` |

## Usage Notes

- **Data type**: float volumes are cast to `int16`; already-integer volumes are
  written as-is.
- **Round-trip**: pairing with [`dicom_to_nifti`](dicom_to_nifti.md) recovers the
  array and geometry (within the `int16` cast).

## Examples

```python
from nidataset.transforms import nifti_to_dicom

series_dir = nifti_to_dicom("scan.nii.gz", "out/")
# Output: out/scan/0000.dcm, out/scan/0001.dcm, ...
```

## Typical Workflow

```python
from nidataset.transforms import nifti_to_dicom, dicom_to_nifti

# Export to DICOM for a viewer, then read back
series_dir = nifti_to_dicom("data/scan.nii.gz", "export/", debug=True)
recovered = dicom_to_nifti(series_dir, "roundtrip/")
print(f"DICOM series: {series_dir}\nRecovered NIfTI: {recovered}")
```
