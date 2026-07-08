<div align="center">

  <!-- headline -->
  <center><h1><img align="center" src="https://raw.githubusercontent.com/GiulioRusso/Ni-Dataset/main/docs/images/logo.png" width=100px> NIfTI Dataset Management</h1></center>

  <!-- PyPI badges -->
  <a href="https://pypi.org/project/nidataset/">
    <img src="https://badge.fury.io/py/nidataset.svg" alt="PyPI version">
  </a>
  <a href="https://pepy.tech/project/nidataset">
    <img src="https://pepy.tech/badge/nidataset" alt="Downloads">
  </a>
  <a href="https://pypi.org/project/nidataset/">
    <img src="https://img.shields.io/pypi/pyversions/nidataset.svg" alt="Python versions">
  </a>
  <a href="https://pypi.org/project/nidataset/">
    <img src="https://img.shields.io/pypi/l/nidataset.svg" alt="License">
  </a>

</div>

<br>

Comprehensive toolkit for NIfTI medical imaging datasets. Extract 2D slices from 3D volumes, validate data integrity (geometry, orientations, NaN), register and resample, compute statistics, generate visualizations, and build detection/segmentation pipelines—all with zero external ML dependencies. <br>

<img align="center" src="https://raw.githubusercontent.com/GiulioRusso/Ni-Dataset/main/docs/images/nidataset.png" width=1000px>

<br>

Extract 2D slices (all views) and build datasets for ML. Preprocess with skull stripping, registration, resampling, MIP. Transform intensity (normalize, window, rescale) and geometry (resample, crop). Analyze (compare, statistics, splits). Visualize (overlays, montages). Validate data integrity (geometry, orientations, NaN, image↔mask coherence) via Python API or CLI (`niqc`). Run quick edits from the shell — rotate, flip, crop, rescale, DICOM↔NIfTI — via the `nii` CLI. No ML dependencies.

### `nii` — quick operations from the terminal

Everyday NIfTI edits without opening Python, mirroring the package API (each command also works on a whole folder):

```bash
nii rotate scan.nii.gz out/ --k 1 --axes 0 1   # lossless 90° rotation
nii flip scans/ out/ --axis 0                  # mirror a whole folder
nii crop-content scan.nii.gz out/ --margin 2   # crop to the foreground box
nii rescale scan.nii.gz out/ --out-min 0 --out-max 255
nii to-dicom scan.nii.gz out/                  # NIfTI → DICOM series
nii from-dicom dicom/case_01/ out/             # DICOM series → NIfTI
```

Geometric ops update the affine in step with the data, so a flip or rotation never corrupts orientation. See the [docs](https://giuliorusso.github.io/Ni-Dataset/) for the full reference.



## ⬇️ Installation and Import
Now, this code is available with PyPI [here](https://pypi.org/project/nidataset/). The package can be installed with:

```bash
pip install nidataset
```

and can be imported as:

```python
import nidataset as nid
```

## 📖 Documentation and Examples

Full documentation: [https://giuliorusso.github.io/Ni-Dataset/](https://giuliorusso.github.io/Ni-Dataset/)

Complete example project: [CT-manager](https://github.com/GiulioRusso/CT-manager) (slice extraction, registration, QC validation, preprocessing pipeline)

## 🚨 Requirements

```bash
nibabel>=5.0.0
numpy>=1.24
opencv-python>=4.7
pandas>=1.5
Pillow>=9.4
scipy>=1.10
SimpleITK>=2.2
scikit-image>=0.19
tqdm>=4.64
```

Install the requirements with:
```bash
pip install -r requirements.txt
```

## 🤝 Contribution
👨‍💻 [Ciro Russo, PhD](https://www.linkedin.com/in/ciro-russo-b14056100/)

## ⚖️ License

MIT License

