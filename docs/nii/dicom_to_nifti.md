---
title: dicom_to_nifti
parent: Quick Operations (nii)
nav_order: 10
---

# `dicom_to_nifti`

Convert a directory of DICOM slices (one series) into a single NIfTI volume.

```python
dicom_to_nifti(
    dicom_dir: str,
    output_path: str,
    debug: bool = False
) -> str
```

## Overview

Reads a DICOM series with SimpleITK's GDCM reader and writes it as a single
`.nii.gz` volume, preserving spacing, origin, and orientation. The output is named
after the DICOM directory.

## Parameters

| Name          | Type   | Default    | Description                                            |
|---------------|--------|------------|--------------------------------------------------------|
| `dicom_dir`   | `str`  | *required* | Directory holding the `.dcm` slices of one series.    |
| `output_path` | `str`  | *required* | Directory where the NIfTI volume is saved.            |
| `debug`       | `bool` | `False`    | If `True`, logs the output path and slice count.      |

## Returns

`str` – Path to the saved NIfTI file.

## Output File

```
<DIRNAME>.nii.gz
```

**Example**: Input directory `dicom/case_01/` → Output `case_01.nii.gz`

## Exceptions

| Exception           | Condition                                              |
|---------------------|--------------------------------------------------------|
| `FileNotFoundError` | `dicom_dir` is not a directory                         |
| `FileNotFoundError` | No DICOM series is found in `dicom_dir`                |

## Usage Notes

- **One series per folder**: the reader uses the series found in `dicom_dir`; keep
  each series in its own directory.
- **Geometry preserved**: spacing, origin, and orientation are carried over from
  the DICOM headers.

## Examples

```python
from nidataset.transforms import dicom_to_nifti

dicom_to_nifti("dicom/case_01/", "out/")
# Output: out/case_01.nii.gz
```

## Typical Workflow

```python
import os
from nidataset.transforms import dicom_to_nifti

# Convert a folder of per-case DICOM series to NIfTI
root = "dicom_cases/"
for case in os.listdir(root):
    case_dir = os.path.join(root, case)
    if os.path.isdir(case_dir):
        dicom_to_nifti(case_dir, "nifti_cases/")
```
