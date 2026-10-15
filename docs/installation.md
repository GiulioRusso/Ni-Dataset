---
title: Installation and Import
nav_order: 2
---

# ⬇️ Installation and Import

nidataset is available on [PyPI](https://pypi.org/project/nidataset/) and needs **Python 3.9 or newer**:

```bash
pip install nidataset
```

and is imported as:

```python
import nidataset as nid
```

## Optional extras

| Extra | Install | Enables |
|-------|---------|---------|
| `yaml` | `pip install "nidataset[yaml]"` | YAML configs for [`nii run`](pipeline/index.md) and `niqc --config`. JSON configs work without it. |
| `dev` | `pip install -e ".[dev]"` (from a clone) | `pytest`, `pytest-cov`, `ruff` for development. |

The full dependency list lives in [`pyproject.toml`](https://github.com/GiulioRusso/Ni-Dataset/blob/main/pyproject.toml).

## External tools

- **FSL** — only for skull stripping ([`skull_CTA`](functions/preprocessing/skull_CTA.md),
  pipeline op `skull_strip`). Install it from the
  [FSL website](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index) and make sure
  `bet` and `fslmaths` are on `PATH`; run from a terminal that sources FSL's profile.
- **Registration templates** — [`register_CTA`](functions/preprocessing/register_CTA.md)
  needs a template brain and brain mask. FSL ships MNI152 templates in
  `$FSLDIR/data/standard/` (e.g. `MNI152_T1_1mm_brain.nii.gz` and
  `MNI152_T1_1mm_brain_mask.nii.gz`).

## Command-line tools

Installing the package also installs two commands:

- `nii` — quick edits and pipelines ([docs](nii/cli.md)).
- `niqc` — quality control ([docs](qc/cli.md)).
