"""In-process tests for the ``nii`` and ``niqc`` command-line entry points."""

import json
import os

import nibabel as nib
import numpy as np
import pytest
from conftest import save_nifti

from nidataset.cli import main as nii
from nidataset.qc.cli import main as niqc


@pytest.fixture
def scan(tmp_path):
    data = np.zeros((8, 7, 6), np.float32)
    data[2:6, 2:5, 1:5] = np.linspace(0, 100, 48, dtype=np.float32).reshape(4, 3, 4)
    return save_nifti(tmp_path / "in" / "scan.nii.gz", data)


def _printed(capsys):
    return capsys.readouterr().out.strip().splitlines()


@pytest.mark.parametrize(
    ("args", "shape"),
    [
        (["rotate", "--k", "1"], (7, 8, 6)),
        (["flip", "--axis", "2"], (8, 7, 6)),
        (["crop", "--bbox", "0", "4", "0", "3", "0", "2"], (4, 3, 2)),
        (["crop-content", "--margin", "1"], (6, 5, 6)),
        (["rescale", "--out-min", "0", "--out-max", "255"], (8, 7, 6)),
    ],
)
def test_nii_volume_commands(scan, tmp_path, capsys, args, shape):
    assert nii([args[0], scan, str(tmp_path / "out"), *args[1:]]) == 0
    (out,) = _printed(capsys)
    assert nib.load(out).shape == shape


def test_nii_folder_input_processes_dataset(scan, tmp_path, capsys):
    save_nifti(tmp_path / "in" / "other.nii.gz", np.ones((3, 3, 3), np.float32))
    assert nii(["flip", os.path.dirname(scan), str(tmp_path / "out")]) == 0
    assert len(_printed(capsys)) == 2


def test_nii_dicom_roundtrip_and_preview(scan, tmp_path, capsys):
    assert nii(["to-dicom", scan, str(tmp_path / "dcm")]) == 0
    (series,) = _printed(capsys)
    assert nii(["from-dicom", series, str(tmp_path / "back")]) == 0
    (back,) = _printed(capsys)
    assert nib.load(back).shape == (8, 7, 6)

    assert nii(["to-dicom", os.path.dirname(scan), str(tmp_path / "dcm2")]) == 0
    assert len(_printed(capsys)) == 1

    assert nii(["preview", scan, str(tmp_path / "png"), "--num-slices", "4", "--cols", "2"]) == 0
    (png,) = _printed(capsys)
    assert png.endswith(".png") and os.path.isfile(png)


@pytest.mark.parametrize("command", ["crop", "preview"])
def test_nii_single_file_commands_reject_folders(scan, tmp_path, capsys, command):
    extra = ["--bbox", "0", "1", "0", "1", "0", "1"] if command == "crop" else []
    assert nii([command, os.path.dirname(scan), str(tmp_path / "o"), *extra]) == 2
    assert "not a folder" in capsys.readouterr().err


def test_nii_missing_file_exits_2(tmp_path, capsys):
    assert nii(["flip", str(tmp_path / "nope.nii.gz"), str(tmp_path / "o")]) == 2
    assert capsys.readouterr().err.startswith("error:")


# niqc


def _pair(tmp_path, shift=0.0):
    img = np.zeros((10, 10, 10), np.float32)
    img[2:8, 2:8, 2:8] = 50
    mask = (img > 0).astype(np.uint8)
    ann = np.zeros_like(mask)
    ann[4:6, 4:6, 4:6] = 1
    affine = np.eye(4)
    shifted = affine.copy()
    shifted[0, 3] = shift
    return (
        save_nifti(tmp_path / "img.nii.gz", img, affine),
        save_nifti(tmp_path / "mask.nii.gz", mask, shifted),
        save_nifti(tmp_path / "ann.nii.gz", ann, affine),
    )


def test_niqc_volume_verbose_text(tmp_path, capsys):
    img, _, _ = _pair(tmp_path)
    assert niqc([img, "--verbose", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "orientation" in out and "ok" in out


def test_niqc_pair_misaligned_is_error_under_strict(tmp_path, capsys):
    img, mask, _ = _pair(tmp_path, shift=3.0)
    assert niqc(["--pair", img, mask, "--no-color"]) == 0
    assert "affine_match_mask" in capsys.readouterr().out
    assert niqc(["--pair", img, mask, "--strict", "--no-color"]) == 1


def test_niqc_triple_json_and_thumbnails(tmp_path, capsys):
    img, mask, ann = _pair(tmp_path)
    report = tmp_path / "r.json"
    assert niqc(["--triple", img, mask, ann, "--json", str(report), "--thumbnails", str(tmp_path / "th")]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["kind"] == "triple"
    assert os.path.isfile(tmp_path / "th" / "qc_thumbnails.png")


def test_niqc_dataset_json_stdout_with_config(tmp_path, capsys):
    _pair(tmp_path)
    cfg = tmp_path / "qc.json"
    cfg.write_text(json.dumps({"expected_orientation": "LPS"}))
    assert niqc([str(tmp_path), "--json", "--config", str(cfg)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_items"] == 3 and data["status"] == "error"  # all LPS-mismatched
    assert niqc([str(tmp_path), "--no-color"]) == 0
    assert "Summary: ✓ 2 ok" in capsys.readouterr().out


def test_niqc_errors(tmp_path, capsys):
    assert niqc([str(tmp_path / "missing.nii.gz")]) == 2
    bad = tmp_path / "qc.json"
    bad.write_text(json.dumps({"nope": 1}))
    img, mask, _ = _pair(tmp_path)
    assert niqc([img, "--config", str(bad)]) == 2
    assert "config error" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        niqc([])
    with pytest.raises(SystemExit):
        niqc([img, "--pair", img, mask])
