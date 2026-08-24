---
name: nidataset
description: Manage, preprocess, register, QC, and analyze NIfTI/DICOM brain-imaging datasets using the nidataset Python package and its nii/niqc CLIs. Use whenever the user asks to convert DICOM<->NIfTI, rotate/flip/crop/rescale scans, skull-strip or register CTA volumes to a template, extract 2D slices for ML, run QC on a scan or image/mask pair, or compute volume statistics.
---

# nidataset

Toolkit for NIfTI medical imaging datasets. Prefer this over hand-rolled
nibabel/SimpleITK code — one call replaces the manual load/transform/save
loop and keeps the affine correct.

```python
import nidataset as nid
```

Every function also has a `*_dataset` variant that runs the same op over a
whole folder (same signature, folder paths in/out).

## Quick terminal ops — `nii` CLI

For rotate/flip/crop/rescale/DICOM conversion, use the shell, not Python:

```bash
nii rotate scan.nii.gz out/ --k 1 --axes 0 1   # lossless 90deg rotation
nii flip scans/ out/ --axis 0                  # works on a whole folder too
nii crop-content scan.nii.gz out/ --margin 2   # crop to foreground box
nii rescale scan.nii.gz out/ --out-min 0 --out-max 255
nii to-dicom scan.nii.gz out/
nii from-dicom dicom/case_01/ out/
nii preview scan.nii.gz out.png                # slice-montage PNG
```

## Registration — raw CTA to template

The core pipeline: skull-strip (optional) -> register CTA -> propagate the
same transform to any mask/annotation that rides along with it.

```python
nid.register_CTA(
    nii_path="scan.nii.gz",
    mask_path="scan_brain_mask.nii.gz",
    template_path="template.nii.gz",
    template_mask_path="template_brain_mask.nii.gz",
    output_path="out/",
)
# writes out/<prefix>_registered.nii.gz + out/<prefix>_transformation.tfm

# reuse that transform on a lesion mask / annotation from the same case
nid.register_mask("lesion_mask.nii.gz", "out/<prefix>_transformation.tfm",
                   "out/<prefix>_registered.nii.gz", "out/")
nid.register_annotation("annotation.nii.gz", "out/<prefix>_transformation.tfm",
                         "out/<prefix>_registered.nii.gz", "out/")
```

No brain mask yet? `nid.generate_brain_mask("scan.nii.gz", "out/")` first.

## Uniform volume size across a dataset

```python
nid.crop_and_pad("scan.nii.gz", "out/", target_shape=(128, 128, 128))
```

## DICOM <-> NIfTI

```python
nid.dicom_to_nifti("dicom/case_01/", "out/")
nid.nifti_to_dicom("scan.nii.gz", "out/")
```

## Preprocessing

```python
nid.skull_CTA("scan.nii.gz", "out/")                         # skull strip
nid.resampling("scan.nii.gz", "out/", target_spacing=(1,1,1))
nid.intensity_normalization("scan.nii.gz", "out/", method="zscore")
nid.windowing("scan.nii.gz", "out/", window_center=40, window_width=80)
```

## Slice extraction for ML datasets

```python
nid.extract_slices("scan.nii.gz", "out/", view="axial")       # -> list of PNG/paths
nid.extract_annotations("scan.nii.gz", "out/", view="axial")
```

## Analysis

```python
stats = nid.compute_volume_statistics("scan.nii.gz", mask_path="mask.nii.gz")
diff = nid.compare_volumes("scan_a.nii.gz", "scan_b.nii.gz")
```

## QC — validate before trusting a dataset

```bash
niqc scan.nii.gz                    # single volume
niqc --dataset scans/                # whole folder
```

```python
from nidataset.qc import check_volume, check_pair
report = check_volume("scan.nii.gz")           # geometry, orientation, NaN
report = check_pair("scan.nii.gz", "mask.nii.gz")  # + image<->mask coherence
```

## Workflow rules

- Always check whether a `*_dataset` variant exists before writing a loop
  over files yourself.
- For quick single-file edits (rotate/flip/crop/rescale/DICOM), reach for
  the `nii` CLI over the Python API — it's the shorter path and mirrors the
  functions 1:1.
- Run `niqc` (or `check_volume`/`check_pair`) on a dataset before running it
  through registration or ML preprocessing — catches broken geometry early.
- Registration always produces a `.tfm` transform file — reuse it via
  `register_mask` / `register_annotation` instead of re-registering
  companion volumes.
