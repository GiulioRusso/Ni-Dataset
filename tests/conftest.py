"""
Shared fixtures: small synthetic volumes only, no real patient data.

``ct_dataset`` reuses the example generator (examples/ct_pipeline/make_synthetic.py),
so the tests also keep that example honest.
"""

import importlib.util
import os
import shutil

import nibabel as nib
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_DIR = os.path.join(ROOT, "examples", "ct_pipeline")

# Non-trivial affine: anisotropic spacing, a flip, and a world-space offset.
AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, -3.0, 0.0, 20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _load_example_generator():
    spec = importlib.util.spec_from_file_location("make_synthetic", os.path.join(EXAMPLE_DIR, "make_synthetic.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_synthetic = _load_example_generator()


def save_nifti(path, data, affine=AFFINE):
    """Save *data* as NIfTI at *path* (parents created) and return the path as str."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


@pytest.fixture
def ct_dataset(tmp_path):
    """Two synthetic cases: ``{images, masks, annotations, template, template_mask, root}``."""
    data = make_synthetic.make(str(tmp_path))
    return {
        "root": str(tmp_path),
        "images": os.path.join(data, "images"),
        "masks": os.path.join(data, "masks"),
        "annotations": os.path.join(data, "annotations"),
        "template": os.path.join(data, "template", "template.nii.gz"),
        "template_mask": os.path.join(data, "template", "template_mask.nii.gz"),
    }


def pytest_collection_modifyitems(config, items):
    if shutil.which("bet") and shutil.which("fslmaths"):
        return
    skip = pytest.mark.skip(reason="FSL (bet, fslmaths) not on PATH")
    for item in items:
        if "fsl" in item.keywords:
            item.add_marker(skip)
