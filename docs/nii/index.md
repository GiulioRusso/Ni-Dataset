---
title: Quick Operations (nii)
nav_order: 5
has_children: true
---

# ⚡ Quick Operations (`nii`)

Fast, everyday NIfTI edits — the kind you would otherwise open a Python REPL for —
available both as a Python API and as the **`nii`** shell command (installed as a
`pyproject.toml` entry point, alongside `niqc`).

Where `niqc` answers *"is this dataset trustworthy?"*, `nii` **changes** volumes:
it is the shell front-end for the geometric and conversion helpers.

## Operations

| Operation | Python | `nii` command |
|-----------|--------|---------------|
| Rotate by 90° steps | [`rotate_volume`](rotate_volume.md) / [`_dataset`](rotate_volume_dataset.md) | `nii rotate` |
| Mirror along an axis | [`flip_volume`](flip_volume.md) / [`_dataset`](flip_volume_dataset.md) | `nii flip` |
| Crop to an explicit box | [`crop_volume`](crop_volume.md) | `nii crop` |
| Crop to the foreground box | [`crop_to_content`](crop_to_content.md) / [`_dataset`](crop_to_content_dataset.md) | `nii crop-content` |
| Linearly rescale intensities | [`rescale_intensity`](rescale_intensity.md) / [`_dataset`](rescale_intensity_dataset.md) | `nii rescale` |
| DICOM series → NIfTI | [`dicom_to_nifti`](dicom_to_nifti.md) | `nii from-dicom` |
| NIfTI → DICOM series | [`nifti_to_dicom`](nifti_to_dicom.md) | `nii to-dicom` |
| Render a montage PNG | [`create_slice_montage`](../functions/visualization/create_slice_montage.md) | `nii preview` |

## Geometry is preserved

Every geometric operation updates the **affine** in step with the data, so a flip
or rotation never silently corrupts left/right or orientation metadata — a marker
voxel keeps its world coordinate before and after. See the
[CLI reference](cli.md) for the shell interface.
