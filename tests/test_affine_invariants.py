"""
World-coordinate invariants for geometric ops, over randomized affines.

For every geometric op, a feature at a known world position must stay there:
an affine not updated in step with the data is the bug class that silently
misaligns images, masks and annotations. Exact ops (rotate/flip/crop) track a
single marker voxel; interpolating ops (resample) track a blob's centroid.
"""

import itertools

import nibabel as nib
import numpy as np
import pytest
from conftest import save_nifti

import nidataset as nid

SHAPE = (14, 12, 10)


def _random_affine(rng):
    """Axis-aligned affine with random spacing, axis flips and offset."""
    spacing = rng.uniform(0.5, 3.0, 3) * rng.choice([-1, 1], 3)
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = rng.uniform(-50, 50, 3)
    return affine


def _world(affine, ijk):
    return affine[:3, :3] @ np.asarray(ijk, float) + affine[:3, 3]


CASES = list(itertools.product(range(4), [(3, 4, 5), (10, 2, 7)]))  # (seed, marker voxel)


def _marker_volume(marker):
    data = np.zeros(SHAPE, dtype=np.float32)
    data[2:6, 2:6, 2:6] = 5.0
    data[marker] = 99.0
    return data


def _marker_world(path):
    img = nib.load(path)
    (idx,) = np.argwhere(np.asarray(img.dataobj) == 99.0)
    return _world(img.affine, idx)


@pytest.mark.parametrize(("seed", "marker"), CASES)
def test_exact_ops_preserve_marker_world(tmp_path, seed, marker):
    affine = _random_affine(np.random.default_rng(seed))
    src = save_nifti(tmp_path / "v.nii.gz", _marker_volume(marker), affine)
    expected = _world(affine, marker)
    outputs = [
        nid.rotate_volume(src, str(tmp_path / "r1"), k=1, axes=(0, 1)),
        nid.rotate_volume(src, str(tmp_path / "r3"), k=3, axes=(1, 2)),
        nid.flip_volume(src, str(tmp_path / "f2"), axis=2),
        nid.crop_volume(src, str(tmp_path / "c"), bbox=(1, 13, 1, 12, 0, 9)),
        nid.crop_and_pad(src, str(tmp_path / "cp"), target_shape=(16, 16, 16)),
    ]
    for out in outputs:
        np.testing.assert_allclose(_marker_world(out), expected, atol=1e-4, err_msg=out)


def _blob_volume(center):
    grid = np.indices(SHAPE).astype(np.float32)
    dist2 = sum((grid[i] - center[i]) ** 2 for i in range(3))
    return (100.0 * np.exp(-dist2 / (2 * 1.5**2))).astype(np.float32)


def _centroid_world(path):
    img = nib.load(path)
    data = np.clip(img.get_fdata(), 0, None)
    ijk = np.array([(np.indices(data.shape)[a] * data).sum() for a in range(3)]) / data.sum()
    return _world(img.affine, ijk), np.abs(np.linalg.eigvals(img.affine[:3, :3])).max()


@pytest.mark.parametrize(("seed", "center"), list(itertools.product(range(4), [(6, 5, 4), (8, 6, 5)])))
def test_resampling_preserves_blob_world(tmp_path, seed, center):
    rng = np.random.default_rng(seed)
    affine = _random_affine(rng)
    src = save_nifti(tmp_path / "v.nii.gz", _blob_volume(center), affine)
    expected = _world(affine, center)

    target = tuple(int(n) for n in rng.integers(7, 20, 3))
    out = nid.resampling(src, str(tmp_path / "rs"), desired_volume=target)
    got, voxel = _centroid_world(out)
    assert nib.load(out).shape == target
    np.testing.assert_allclose(got, expected, atol=0.5 * voxel)

    ref_affine = _random_affine(rng)
    ref_affine[:3, 3] = expected - ref_affine[:3, :3] @ np.array(SHAPE) / 2  # keep the blob in view
    ref = save_nifti(tmp_path / "ref.nii.gz", np.zeros(SHAPE, np.float32), ref_affine)
    out = nid.resample_to_reference(src, ref, str(tmp_path / "rr"))
    got, voxel = _centroid_world(out)
    np.testing.assert_allclose(nib.load(out).affine, ref_affine, atol=1e-4)
    np.testing.assert_allclose(got, expected, atol=0.5 * voxel)
