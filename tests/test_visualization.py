"""
Regression tests for nidataset.visualization.

Synthetic in-memory NIfTI volumes only.
"""

import numpy as np
import nibabel as nib

import nidataset as nid


def _save(tmp_path, data, name):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return str(path)


def test_overlay_mask_runs_on_modern_matplotlib(tmp_path):
    # _get_colormap used matplotlib.cm.get_cmap, removed in matplotlib >= 3.9;
    # the overlay must still produce one image per slice.
    rng = np.random.default_rng(0)
    vol = (rng.random((8, 8, 5)) * 100).astype(np.float32)
    mask = np.zeros((8, 8, 5), dtype=np.uint8)
    mask[2:6, 2:6, 1:4] = 1

    vp = _save(tmp_path, vol, "v.nii.gz")
    mp = _save(tmp_path, mask, "m.nii.gz")

    paths = nid.overlay_mask_on_volume(vp, mp, str(tmp_path / "out"), view="axial")
    assert len(paths) == 5
    for p in paths:
        assert nib.os.path.isfile(p)
