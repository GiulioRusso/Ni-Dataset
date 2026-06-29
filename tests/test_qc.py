"""
Tests for nidataset.qc.

All NIfTI volumes are synthetic and built in-memory with known affines — no
dependency on real data. Each test targets one of the silent bugs the module
exists to catch: LAS vs RAS orientation, a mask affine shifted from the image,
empty / out-of-mask annotations, anisotropic spacing, NaN, int64, all-black
slices, shape mismatch and the 4D case.
"""

import json
import subprocess
import sys

import numpy as np
import nibabel as nib

import nidataset as nid
from nidataset.qc import (
    check_volume,
    check_pair,
    check_triple,
    check_dataset,
    QCConfig,
    to_json,
    OK,
    WARNING,
    ERROR,
)

RAS = np.diag([1.0, 1.0, 1.0, 1.0])  # identity -> ('R', 'A', 'S')


def _status_of(report, name):
    for r in report.results:
        if r.name == name:
            return r.status
    raise AssertionError(f"check '{name}' not found in {[r.name for r in report.results]}")


def _save(tmp_path, name, data, affine=RAS, header=None):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(data, affine, header), str(path))
    return str(path)


def _block(shape=(20, 20, 20), lo=4, hi=16, value=1, dtype=np.uint8):
    a = np.zeros(shape, dtype=dtype)
    a[lo:hi, lo:hi, lo:hi] = value
    return a


# Subpackage wiring

def test_qc_exposed_as_nid_qc():
    assert hasattr(nid, "qc")
    assert nid.qc.check_volume is check_volume


# Single volume — geometry

def test_orientation_las_flagged_against_ras(tmp_path):
    las = np.diag([-1.0, 1.0, 1.0, 1.0])  # flips R->L  => ('L', 'A', 'S')
    path = _save(tmp_path, "las.nii.gz", np.random.rand(10, 10, 10).astype(np.float32), las)
    rep = check_volume(path, QCConfig(expected_orientation="RAS"))
    assert _status_of(rep, "orientation") == ERROR
    assert rep.meta["orientation"] == "LAS"


def test_ras_passes_orientation(tmp_path):
    path = _save(tmp_path, "ras.nii.gz", np.random.rand(10, 10, 10).astype(np.float32), RAS)
    rep = check_volume(path, QCConfig(expected_orientation="RAS"))
    assert _status_of(rep, "orientation") == OK


def test_singular_affine_flagged(tmp_path):
    # A truly zero column can't be written by nibabel (qform decomposition fails),
    # so use a near-collapsed k axis: det is below affine_atol => still "singular".
    sing = np.diag([1.0, 1.0, 1e-6, 1.0])  # det(direction) = 1e-6 < default atol 1e-4
    path = _save(tmp_path, "sing.nii.gz", np.random.rand(8, 8, 8).astype(np.float32), sing)
    rep = check_volume(path)
    assert _status_of(rep, "affine_nonsingular") == ERROR


def test_anisotropic_spacing_warns(tmp_path):
    aniso = np.diag([1.0, 1.0, 5.0, 1.0])  # 5x along k
    path = _save(tmp_path, "aniso.nii.gz", np.random.rand(8, 8, 8).astype(np.float32), aniso)
    rep = check_volume(path, QCConfig(isotropy_tol=0.05))
    assert _status_of(rep, "spacing_isotropy") == WARNING


# Single volume — data

def test_nan_flagged(tmp_path):
    data = np.random.rand(8, 8, 8).astype(np.float32)
    data[0, 0, 0] = np.nan
    path = _save(tmp_path, "nan.nii.gz", data)
    rep = check_volume(path)
    assert _status_of(rep, "finite_values") == ERROR


def test_int64_dtype_warns(tmp_path):
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.int64)
    data = _block(dtype=np.int64)
    path = _save(tmp_path, "i64.nii.gz", data, RAS, hdr)
    rep = check_volume(path)
    assert _status_of(rep, "dtype") == WARNING


def test_all_zero_volume_flagged(tmp_path):
    path = _save(tmp_path, "zero.nii.gz", np.zeros((8, 8, 8), np.float32))
    rep = check_volume(path)
    assert _status_of(rep, "constant_volume") == ERROR


def test_empty_border_slices_warn(tmp_path):
    # Foreground only in the centre -> many all-black border slices on every axis.
    data = np.zeros((20, 20, 20), np.float32)
    data[9:11, 9:11, 9:11] = 1.0
    path = _save(tmp_path, "padded.nii.gz", data)
    rep = check_volume(path, QCConfig(max_empty_slice_fraction=0.5))
    assert _status_of(rep, "empty_slices") == WARNING


# Single volume — file / shape

def test_4d_volume_reported(tmp_path):
    data = np.random.rand(8, 8, 8, 3).astype(np.float32)
    path = _save(tmp_path, "ts.nii.gz", data)
    rep = check_volume(path)
    assert _status_of(rep, "dimensionality") == WARNING
    assert rep.meta["ndim"] == 4


# Pair / triple

def test_pair_matching_ok(tmp_path):
    img = _save(tmp_path, "img.nii.gz", np.random.rand(20, 20, 20).astype(np.float32))
    mask = _save(tmp_path, "mask.nii.gz", _block())
    rep = check_pair(img, mask)
    assert rep.status == OK


def test_pair_affine_shift_flagged(tmp_path):
    shifted = RAS.copy()
    shifted[:3, 3] = [0, 0, 3]  # 3 mm world shift, identical shape
    img = _save(tmp_path, "img.nii.gz", np.random.rand(20, 20, 20).astype(np.float32), RAS)
    mask = _save(tmp_path, "mask.nii.gz", _block(), shifted)
    rep = check_pair(img, mask)
    assert _status_of(rep, "affine_match_mask") == ERROR
    val = next(r.value for r in rep.results if r.name == "affine_match_mask")
    assert val["translation_mm"] == 3.0


def test_pair_shape_mismatch_flagged(tmp_path):
    img = _save(tmp_path, "img.nii.gz", np.random.rand(20, 20, 20).astype(np.float32))
    mask = _save(tmp_path, "mask.nii.gz", _block(shape=(20, 20, 18)))
    rep = check_pair(img, mask)
    assert _status_of(rep, "shape_match_mask") == ERROR


def test_triple_empty_annotation_flagged(tmp_path):
    img = _save(tmp_path, "img.nii.gz", np.random.rand(20, 20, 20).astype(np.float32))
    mask = _save(tmp_path, "mask.nii.gz", _block())
    ann = _save(tmp_path, "ann.nii.gz", np.zeros((20, 20, 20), np.uint8))
    rep = check_triple(img, mask, ann)
    assert _status_of(rep, "nonempty_annotation") == ERROR


def test_triple_annotation_outside_mask_flagged(tmp_path):
    img = _save(tmp_path, "img.nii.gz", np.random.rand(20, 20, 20).astype(np.float32))
    mask = _save(tmp_path, "mask.nii.gz", _block(lo=2, hi=6))     # mask in one corner
    ann = _save(tmp_path, "ann.nii.gz", _block(lo=14, hi=18))     # annotation in another
    rep = check_triple(img, mask, ann)
    assert _status_of(rep, "containment") == ERROR
    frac = next(r.value for r in rep.results if r.name == "containment")
    assert frac == 1.0


def test_triple_labels_outside_allowed_flagged(tmp_path):
    img = _save(tmp_path, "img.nii.gz", np.random.rand(20, 20, 20).astype(np.float32))
    mask = _save(tmp_path, "mask.nii.gz", _block())
    ann = _save(tmp_path, "ann.nii.gz", _block(lo=8, hi=12, value=7))  # label 7, not binary
    rep = check_triple(img, mask, ann)
    assert _status_of(rep, "labels_annotation") == ERROR


# Dataset

def test_dataset_orientation_distribution_and_outliers(tmp_path):
    for i in range(3):
        _save(tmp_path, f"ras_{i}.nii.gz", np.random.rand(10, 10, 10).astype(np.float32), RAS)
    las = np.diag([-1.0, 1.0, 1.0, 1.0])
    _save(tmp_path, "las_0.nii.gz", np.random.rand(10, 10, 10).astype(np.float32), las)

    ds = check_dataset(str(tmp_path))
    assert ds.distributions["orientation"] == {"RAS": 3, "LAS": 1}
    assert any("las_0" in p for p in ds.distributions["outliers"]["orientation"])
    assert len(ds.items) == 4


def test_dataset_csv_triples(tmp_path):
    img = _save(tmp_path, "img.nii.gz", np.random.rand(16, 16, 16).astype(np.float32))
    mask = _save(tmp_path, "mask.nii.gz", _block(shape=(16, 16, 16), lo=3, hi=13))
    ann = _save(tmp_path, "ann.nii.gz", _block(shape=(16, 16, 16), lo=6, hi=9))
    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(f"image,mask,annotation\n{img},{mask},{ann}\n")

    ds = check_dataset(str(csv_path))
    assert len(ds.items) == 1
    assert ds.items[0].kind == "triple"


# Serialization

def test_report_json_roundtrips(tmp_path):
    img = _save(tmp_path, "img.nii.gz", np.random.rand(10, 10, 10).astype(np.float32))
    rep = check_volume(img)
    text = to_json(rep)
    data = json.loads(text)
    assert data["kind"] == "volume"
    assert "results" in data and isinstance(data["results"], list)


# Config

def test_config_load_json(tmp_path):
    cfg_path = tmp_path / "qc.json"
    cfg_path.write_text(json.dumps({"expected_orientation": "ras", "affine_atol": 0.01}))
    cfg = QCConfig.load(str(cfg_path))
    assert cfg.expected_orientation == "RAS"
    assert cfg.affine_atol == 0.01


def test_config_unknown_key_rejected(tmp_path):
    cfg_path = tmp_path / "qc.json"
    cfg_path.write_text(json.dumps({"nope": 1}))
    try:
        QCConfig.load(str(cfg_path))
    except ValueError:
        return
    raise AssertionError("unknown key should raise ValueError")


# Thumbnails

def test_thumbnail_grid_written(tmp_path):
    img = _save(tmp_path, "img.nii.gz", np.random.rand(20, 20, 20).astype(np.float32))
    mask = _save(tmp_path, "mask.nii.gz", _block())
    out = str(tmp_path / "grid.png")
    nid.qc.thumbnail_grid([{"image": img, "mask": mask}], out)
    assert (tmp_path / "grid.png").exists()


# CLI

def _run_cli(args):
    return subprocess.run([sys.executable, "-m", "nidataset.qc.cli", *args],
                          capture_output=True, text=True)


def test_cli_strict_exit_code_on_error(tmp_path):
    data = np.random.rand(8, 8, 8).astype(np.float32)
    data[0, 0, 0] = np.nan
    path = _save(tmp_path, "nan.nii.gz", data)
    assert _run_cli([path, "--strict", "--no-color"]).returncode == 1
    # Without --strict the same error must not fail the process.
    assert _run_cli([path, "--no-color"]).returncode == 0


def test_cli_json_is_valid(tmp_path):
    path = _save(tmp_path, "img.nii.gz", np.random.rand(8, 8, 8).astype(np.float32))
    out = _run_cli([path, "--json"])
    assert out.returncode == 0
    parsed = json.loads(out.stdout)
    assert parsed["kind"] == "volume"


def test_cli_thumbnails(tmp_path):
    _save(tmp_path, "img.nii.gz", np.random.rand(16, 16, 16).astype(np.float32))
    thumbs = tmp_path / "thumbs"
    out = _run_cli([str(tmp_path), "--thumbnails", str(thumbs), "--no-color"])
    assert out.returncode == 0
    assert (thumbs / "qc_thumbnails.png").exists()
