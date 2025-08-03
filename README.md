<div align="center">

  <!-- headline -->
  <center><h1><img align="center" src="./docs/images/logo.png" width=100px> NIfTI Dataset Management</h1></center>

  <!-- PyPI badge -->
  <a href="https://pypi.org/project/nidataset/">
    <img src="https://badge.fury.io/py/nidataset.svg" alt="PyPI version">
  </a>

</div>

<br>

This package provides a set of utilities for handling NIfTI datasets, including slice extraction, volume manipulation, and various utility functions to facilitate the processing of medical imaging data. <br>

<img align="center" src="./docs/images/nidataset.png" width=1000px>

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

## 🚨 Requirements

```bash
Python>=3.8.0
Pillow>=9.4.0
nibabel>=5.1.0
numpy>=1.24.2
scikit-image>=0.19.3
pandas>=1.5.3
SimpleITK>=2.2.1
scipy>=1.10.0
tqdm>=4.67.1
```

Install the requirements with:
```bash
pip install -r requirements.txt
```

## 📂 Project Organization

The package consists of the following Python modules:
```bash
.
├── nidataset/                # The NIfTI dataset management package folder
│   ├── Draw.py               # Functions for drawing and manipulating bounding boxes on NIfTI images.
│   ├── Preprocessing.py      # Functions for preprocessing pipelines on NIfTI images.
│   ├── Slices.py             # Functions for extracting slices and annotations from NIfTI files.
│   ├── Utility.py            # Utility functions for dataset information statistics.
│   └── Volume.py             # Functions for NIfTI volume transformations and modifications.
│
├── example.py                # The script that demonstrates usage of the package.
│
├── dataset/                  # Example dataset folder
│   ├── toy-CTA.nii.gz        # Example NIfTI file.
│   └── toy-annotation.nii.gz # Example annotation file.
│
└── output/                   # Folder for output results
```

Run the application example with:

```bash
python3 example.py
```

This code will extract the slices and the annotations from a toy CTA and annotation bounding box. Then axial and coronal views are shifted.

## 📦 Package documentation

Package documentation is available [here](https://giuliorusso.github.io/Ni-Dataset/).

## 🤝 Contribution
👨‍💻 [Ciro Russo, PhD](https://www.linkedin.com/in/ciro-russo-b14056100/)

## ⚖️ License

MIT License

