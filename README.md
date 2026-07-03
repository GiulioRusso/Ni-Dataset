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

## 📚 Features

- **Slicing**: Extract 2D slices (axial, coronal, sagittal) with flexible modes and statistics.
- **Volume processing**: Crop, pad, generate brain masks, extract bounding boxes, build heatmaps.
- **Preprocessing**: Skull stripping (FSL), registration to templates, MIP projections, resampling to target volume.
- **Transforms**: Intensity normalization, windowing (CT presets), spatial resampling.
- **Analysis**: Compare volumes, compute statistics (volume, surface area), split datasets for train/val/test.
- **Visualization**: Overlay masks on volumes, create slice montages.
- **Quality Control**: Validate geometry (affine, orientation, spacing), data integrity (NaN, dtype), and image↔mask coherence. Catch silent bugs (orientation mismatches, affine shifts, anisotropic spacing) that poison training. Python API + CLI (`niqc`).
- **Utilities**: Dataset metadata extraction, drawing annotations.

<br>

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

