"""
Tests for nidataset.spatial (rotate/flip/crop) and the new transforms
(rescale, DICOM conversion).

The core invariant for geometric ops: a marker voxel keeps its *world*
coordinate after the operation. That is what catches an affine that was not
updated in step with the data.
"""

import numpy as np
import nibabel as nib
import pytest

import nidataset as nid


# Non-trivial affine: anisotropic spacing, a flip, and a world offset.
AFFINE = np.array([
    [2.0, 0.0, 0.0, 10.0],
    [0.0, -3.0, 0.0, 20.0],
    [0.0, 0.0, 4.0, 30.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _world(affine, ijk):
    return affine[:3, :3] @ np.asarray(ijk, float) + affine[:3, 3]


def _marker_volume(shape=(9, 7, 5)):
    """A volume with one unique max voxel, used as a world-space marker."""
    data = np.zeros(shape, dtype=np.float32)
    data[6, 2, 3] = 99.0  # unique marker
    data[1:4, 1:5, 1:4] = 10.0  # a foreground blob offset from the borders
    return data


def _save(tmp_path, data, affine=AFFINE, name="vol.nii.gz"):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return str(path)


def _load(path):
    img = nib.load(path)
    return np.asarray(img.dataobj), img.affine


def _marker_world(data, affine, value=99.0):
    idx = np.argwhere(data == value)
    assert len(idx) == 1, "marker must remain unique"
    return _world(affine, idx[0])


def test_flip_preserves_marker_world(tmp_path):
    data = _marker_volume()
    src = _save(tmp_path, data)
    before = _marker_world(data, AFFINE)
    for axis in (0, 1, 2):
        out = nid.flip_volume(src, str(tmp_path / f"f{axis}"), axis=axis)
        od, oa = _load(out)
        assert np.allclose(_marker_world(od, oa), before)


def test_rotate_preserves_marker_world(tmp_path):
    data = _marker_volume()
    src = _save(tmp_path, data)
    before = _marker_world(data, AFFINE)
    for k in (1, 2, 3, -1):
        out = nid.rotate_volume(src, str(tmp_path / f"r{k}"), k=k, axes=(0, 1))
        od, oa = _load(out)
        assert np.allclose(_marker_world(od, oa), before)


def test_rotate_k4_is_identity(tmp_path):
    data = _marker_volume()
    src = _save(tmp_path, data)
    out = nid.rotate_volume(src, str(tmp_path / "r4"), k=4, axes=(1, 2))
    od, oa = _load(out)
    assert od.shape == data.shape
    assert np.allclose(od, data)
    assert np.allclose(oa, AFFINE)


def test_crop_preserves_marker_world_and_shape(tmp_path):
    data = _marker_volume()
    src = _save(tmp_path, data)
    before = _marker_world(data, AFFINE)
    out = nid.crop_volume(src, str(tmp_path / "c"), bbox=(1, 8, 1, 6, 1, 5))
    od, oa = _load(out)
    assert od.shape == (7, 5, 4)
    assert np.allclose(_marker_world(od, oa), before)


def test_crop_rejects_out_of_range(tmp_path):
    src = _save(tmp_path, _marker_volume())
    with pytest.raises(ValueError):
        nid.crop_volume(src, str(tmp_path / "c"), bbox=(0, 100, 0, 6, 0, 5))


def test_crop_to_content_trims_border(tmp_path):
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[2:5, 3:8, 1:4] = 7.0
    src = _save(tmp_path, data)
    before = _world(AFFINE, [2, 3, 1])  # world coord of the blob corner
    out = nid.crop_to_content(src, str(tmp_path / "cc"))
    od, oa = _load(out)
    assert od.shape == (3, 5, 3)
    assert np.allclose(_world(oa, [0, 0, 0]), before)
    assert (od == 7.0).all()


def test_crop_to_content_margin_and_empty(tmp_path):
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[4:6, 4:6, 4:6] = 1.0
    src = _save(tmp_path, data)
    out = nid.crop_to_content(src, str(tmp_path / "m"), margin=1)
    od, _ = _load(out)
    assert od.shape == (4, 4, 4)  # 2 + 1 margin each side

    empty = _save(tmp_path, np.zeros((5, 5, 5), np.float32), name="empty.nii.gz")
    with pytest.raises(ValueError):
        nid.crop_to_content(empty, str(tmp_path / "e"))


def test_rescale_hits_target_range(tmp_path):
    data = (np.random.rand(8, 8, 8).astype(np.float32) * 400) - 100
    src = _save(tmp_path, data)
    out = nid.rescale_intensity(src, str(tmp_path / "rs"), out_min=0, out_max=255)
    od, _ = _load(out)
    assert np.isclose(od.min(), 0.0, atol=1e-3)
    assert np.isclose(od.max(), 255.0, atol=1e-3)


def test_rescale_clips_to_input_window(tmp_path):
    data = np.linspace(0, 100, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    src = _save(tmp_path, data)
    out = nid.rescale_intensity(src, str(tmp_path / "rw"), out_min=0, out_max=1,
                                in_min=25, in_max=75)
    od, _ = _load(out)
    assert od.min() == 0.0 and od.max() == 1.0  # values outside [25,75] clamp


def test_dicom_roundtrip(tmp_path):
    data = (np.random.rand(12, 10, 6) * 500).astype(np.int16).astype(np.float32)
    src = _save(tmp_path, data)
    series_dir = nid.nifti_to_dicom(src, str(tmp_path / "dcm"))
    recovered = nid.dicom_to_nifti(series_dir, str(tmp_path / "back"))
    rd, ra = _load(recovered)
    assert rd.shape == data.shape
    assert np.allclose(rd, data, atol=1.0)
    # geometry preserved (spacing from AFFINE diagonal magnitudes)
    spacing = np.linalg.norm(ra[:3, :3], axis=0)
    assert np.allclose(sorted(spacing), sorted([2.0, 3.0, 4.0]), atol=1e-3)
