---
title: nii CLI
parent: Quick Operations (nii)
nav_order: 12
---

# `nii` — command-line interface

```bash
nii <command> PATH OUTPUT_DIR [options]
```

For `rotate`, `flip`, `crop-content`, and `rescale`, `PATH` may be a **single
NIfTI file or a folder** (auto-detected; a folder is processed as a dataset).
`crop` takes a single file; `from-dicom` takes a DICOM series folder. Each
command prints the output path(s) it wrote. Exit code `0` on success, `2` on a
usage / runtime error.

## Commands

| Command | Purpose | Key options |
|---------|---------|-------------|
| `rotate` | Rotate by `k`×90° in a voxel plane (lossless) | `--k INT` (default 1), `--axes A B` (default `0 1`) |
| `flip` | Mirror along a voxel axis | `--axis {0,1,2}` (default 0) |
| `crop` | Crop to an explicit voxel box | `--bbox X0 X1 Y0 Y1 Z0 Z1` (required) |
| `crop-content` | Crop to the minimal foreground box | `--threshold FLOAT`, `--margin INT` |
| `rescale` | Linearly rescale intensities | `--out-min`, `--out-max`, `--in-min`, `--in-max` |
| `to-dicom` | Convert NIfTI → DICOM series | `--series-description STR` |
| `from-dicom` | Convert DICOM series folder → NIfTI | — |
| `preview` | Render a slice-montage PNG of a volume | `--view {axial,coronal,sagittal}`, `--num-slices INT`, `--cols INT` |

## Examples

```bash
# Rotate 90° in the axial plane, save into out/
nii rotate scan.nii.gz out/ --k 1 --axes 0 1

# Mirror every volume in a folder along the first axis
nii flip scans/ out/ --axis 0

# Crop to an explicit voxel box
nii crop scan.nii.gz out/ --bbox 10 100 10 100 5 60

# Trim empty borders, keeping a 2-voxel margin
nii crop-content scan.nii.gz out/ --margin 2

# Rescale intensities to 0–255 (e.g. for 8-bit export)
nii rescale scan.nii.gz out/ --out-min 0 --out-max 255

# Convert to a DICOM series and back
nii to-dicom scan.nii.gz out/
nii from-dicom dicom/case_01/ out/

# Render a montage PNG to eyeball a result
nii preview out/scan_content.nii.gz out/ --view axial --num-slices 16
```

Output filenames carry an operation suffix: `_rot<k>`, `_flip<axis>`,
`_cropped`, `_content`, `_rescaled`. `to-dicom` writes a folder named after the
input volume containing one `.dcm` per slice.
