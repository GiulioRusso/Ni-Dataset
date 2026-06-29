"""
Regression tests for nidataset.swap_nifti_views.

Synthetic in-memory NIfTI volumes only (small arrays, known affines).
"""

import numpy as np
import nibabel as nib

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


def test_roundtrip_restores_data_and_affine(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 100, size=(6, 8, 10)).astype(np.float32)
    nii_path = _save(tmp_path, data)

    swapped = nid.swap_nifti_views(nii_path, str(tmp_path / "s1"),
                                   source_view="axial", target_view="coronal")
    back = nid.swap_nifti_views(swapped, str(tmp_path / "s2"),
                                source_view="coronal", target_view="axial")

    back_img = nib.load(back)
    np.testing.assert_array_equal(back_img.get_fdata(), data)
    np.testing.assert_allclose(back_img.affine, AFFINE, atol=1e-6)


def test_preserves_world_position(tmp_path):
    # A marker voxel's world coordinate must be identical before and after a
    # single swap; the old affine permuted matrix rows and ignored the rotation.
    data = np.zeros((6, 8, 10), dtype=np.float32)
    marker_ijk = (1, 2, 3)
    data[marker_ijk] = 42.0
    nii_path = _save(tmp_path, data)

    world_before = _voxel_to_world(AFFINE, marker_ijk)

    out = nid.swap_nifti_views(nii_path, str(tmp_path / "s"),
                               source_view="axial", target_view="sagittal")
    img = nib.load(out)
    found = np.argwhere(img.get_fdata() == 42.0)
    assert len(found) == 1
    world_after = _voxel_to_world(img.affine, found[0])

    np.testing.assert_allclose(world_after, world_before, atol=1e-6)
