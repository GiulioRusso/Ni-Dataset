"""
Regression tests for nidataset.crop_and_pad.

Synthetic in-memory NIfTI volumes only (small arrays, known affines).
"""

import numpy as np
import nibabel as nib
import pytest

import nidataset as nid


# Non-trivial affine: anisotropic spacing, a flip, and a world-space offset.
AFFINE = np.array([
    [2.0, 0.0, 0.0, 10.0],
    [0.0, -3.0, 0.0, 20.0],
    [0.0, 0.0, 4.0, 30.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _voxel_to_world(affine, ijk):
    ijk = np.asarray(ijk, dtype=np.float64)
    return affine[:3, :3] @ ijk + affine[:3, 3]


def _save(tmp_path, data, affine=AFFINE, name="vol.nii.gz"):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return str(path)


def test_odd_crop_yields_exact_target_shape(tmp_path):
    # bbox is 11 voxels on X (odd over-size vs target); the old symmetric-crop
    # code left an extra voxel, so the output must match target exactly now.
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[2:13, 2:10, 2:10] = 50.0  # bbox shape (11, 8, 8)
    nii_path = _save(tmp_path, data)

    out = nid.crop_and_pad(nii_path, str(tmp_path / "out"), target_shape=(8, 8, 8))
    shape = nib.load(out).shape
    assert shape == (8, 8, 8), f"expected exact target shape, got {shape}"


def test_preserves_world_position(tmp_path):
    # A marker voxel must keep its world coordinate after crop+pad. The old
    # affine update ignored the centered-crop offset, shifting the volume.
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[2:14, 2:14, 2:14] = 50.0  # bbox shape (12, 12, 12)
    marker_ijk = (8, 7, 6)
    data[marker_ijk] = 99.0
    nii_path = _save(tmp_path, data)

    world_before = _voxel_to_world(AFFINE, marker_ijk)

    out = nid.crop_and_pad(nii_path, str(tmp_path / "out"), target_shape=(6, 6, 6))
    img = nib.load(out)
    found = np.argwhere(img.get_fdata() == 99.0)
    assert len(found) == 1, "marker should survive the crop"
    world_after = _voxel_to_world(img.affine, found[0])

    np.testing.assert_allclose(world_after, world_before, atol=1e-6)
