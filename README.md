<div align="center">

  <!-- headline -->
  <center><h1><img align="center" src="https://raw.githubusercontent.com/GiulioRusso/Ni-Dataset/main/docs/images/logo.png" width=100px> NIfTI Dataset Management</h1></center>

  <a href="https://pypi.org/project/nidataset/"><img src="https://img.shields.io/pypi/v/nidataset.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/nidataset/"><img src="https://img.shields.io/pypi/pyversions/nidataset.svg" alt="Python versions"></a>
  <a href="https://github.com/GiulioRusso/Ni-Dataset/actions/workflows/ci.yml"><img src="https://github.com/GiulioRusso/Ni-Dataset/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/GiulioRusso/Ni-Dataset/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/coverage-%E2%89%A570%25-brightgreen" alt="Coverage (CI gate)"></a>
  <a href="https://github.com/GiulioRusso/Ni-Dataset/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/nidataset.svg" alt="License"></a>
  <a href="https://pepy.tech/project/nidataset"><img src="https://static.pepy.tech/badge/nidataset" alt="Downloads"></a>

</div>

<br>

**NIfTI dataset toolkit for CT/MRI deep-learning pipelines: slice extraction, registration, QC, preprocessing — no ML dependencies.**

```bash
pip install nidataset            # add [yaml] for YAML pipeline / QC configs
```

```python
import nidataset as nid

print(nid.qc.check_dataset("scans/").status)  # orientation, spacing, NaN, ... before anything else
nid.resampling_dataset("scans/", "resampled/", desired_volume=(224, 224, 128), saving_mode="folder")
nid.extract_slices_dataset("resampled/", "slices/", view="axial", saving_mode="view")
nid.extract_annotations_dataset("labels/", "annotations/", view="axial", saving_mode="view")
```

Or chain the same steps in one config and run it from the shell ([example](https://github.com/GiulioRusso/Ni-Dataset/tree/main/examples/ct_pipeline)):

```bash
cd examples/ct_pipeline
python make_synthetic.py         # tiny synthetic dataset, no patient data
nii run pipeline.yaml            # QC -> slices -> registration -> ... + run_manifest.json
```

<img align="center" src="https://raw.githubusercontent.com/GiulioRusso/Ni-Dataset/main/docs/images/nidataset.png" width=1000px>

## ✨ What's inside

- **Slices for ML**: extract 2D slices and matching annotations (center / box / radius) from 3D volumes, in any view, with optional padding.
- **Preprocessing**: skull stripping (FSL), registration to a template and propagation of the transform to masks and annotations, resampling, MIP.
- **Transforms**: intensity normalization, CT windowing presets, rescaling, resample-to-reference, DICOM ↔ NIfTI.
- **Geometry**: rotate, flip, crop, crop-to-content, crop-and-pad — all keep the affine in step with the data (tested on world coordinates).
- **QC**: geometry, orientations, NaN/Inf, empty slices, image ↔ mask ↔ annotation coherence, via Python or `niqc`.
- **Analysis and visualization**: volume comparison (Dice, Hausdorff, …), statistics, dataset splits, overlays, montages.
- **CLIs**: `nii` for quick edits (`nii flip scans/ out/ --axis 0`) and config-driven pipelines (`nii run`), `niqc` for QC.

## 📖 Documentation and examples

- Full documentation: [giuliorusso.github.io/Ni-Dataset](https://giuliorusso.github.io/Ni-Dataset/)
- End-to-end CT pipeline: [`examples/ct_pipeline/`](https://github.com/GiulioRusso/Ni-Dataset/tree/main/examples/ct_pipeline)

Dependencies are declared in [`pyproject.toml`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/pyproject.toml). Skull stripping additionally needs a local [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index) installation.

## 📚 Used in

<!-- TODO(Giulio): list the publications that used nidataset, with DOIs. -->

If you use nidataset in your research, please cite it ([`CITATION.cff`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/CITATION.cff)).

## 🤝 Contributing

See [`CONTRIBUTING.md`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/CONTRIBUTING.md) and the [`CHANGELOG.md`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/CHANGELOG.md).

Authors: Giulio Russo, [Ciro Russo, PhD](https://www.linkedin.com/in/ciro-russo-b14056100/)

## ⚖️ License

MIT License
