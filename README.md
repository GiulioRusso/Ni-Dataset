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

This package provides a set of utilities for handling NIfTI datasets, including slice extraction, volume manipulation, and various utility functions to facilitate the processing of medical imaging data. <br>

<img align="center" src="https://raw.githubusercontent.com/GiulioRusso/Ni-Dataset/main/docs/images/nidataset.png" width=1000px>

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

## 📦 Package documentation

Package documentation is available [here](https://giuliorusso.github.io/Ni-Dataset/).

A complete project example that use `nidataset` is available [here](https://github.com/GiulioRusso/CT-manager)

## 🩺 Quality Control (`qc`)

The `nidataset.qc` sub-module validates the *geometric coherence* of NIfTI datasets
for detection/segmentation. It answers a question no viewer does: **is this dataset
trustworthy, or is something silently poisoning training?** It catches the bugs that
never raise — unexpected orientation (LAS vs RAS), a mask shifted a few voxels from
its image, empty annotations, anisotropic spacing, all-black border slices,
non-portable `int64` data, NaN/inf — and reports them as inspectable, serializable
objects.

**Python API** (`nid.qc.<fn>`, every function returns a report):

```python
import nidataset as nid

# Single volume
rep = nid.qc.check_volume("scan.nii.gz")
print(rep.status)                       # 'ok' | 'warning' | 'error'
print([r.name for r in rep.issues()])   # only the problems

# Image <-> mask <-> annotation coherence (the high-value path)
rep = nid.qc.check_pair("ct.nii.gz", "brain_mask.nii.gz")
rep = nid.qc.check_triple("ct.nii.gz", "brain.nii.gz", "lesion.nii.gz")

# Whole dataset + custom thresholds + JSON export
cfg = nid.qc.QCConfig(expected_orientation="RAS", affine_atol=1e-3)
ds = nid.qc.check_dataset("scans/", config=cfg)
print(ds.distributions["orientation"])  # e.g. {'RAS': 287, 'LAS': 13}
nid.qc.to_json(ds, "qc_report.json")
```

**CLI** (`niqc`, auto-detects file / folder / CSV of triples):

```bash
niqc scan.nii.gz                      # single volume, coloured report
niqc scans/ --strict                  # fail CI (exit 1) on any error
niqc triples.csv --json report.json   # CSV manifest -> structured JSON
niqc --pair ct.nii.gz brain.nii.gz    # explicit image/mask
niqc --triple ct.nii.gz brain.nii.gz lesion.nii.gz --thumbnails qc/
niqc scans/ --config qc.yaml          # custom thresholds
```

All thresholds (orientation, affine/isotropy tolerances, spacing/intensity ranges,
empty-slice definition, allowed labels, containment) live in `QCConfig` and can be
loaded from a config file. See **[`qc.example.yaml`](docs/qc/qc.example.yaml)** (commented,
needs the optional `pyyaml` extra) or **[`qc.example.json`](docs/qc/qc.example.json)**
(no extra dependency). Design rationale and default values are in
**[`QC_DESIGN.md`](docs/QC_DESIGN.md)**.

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

