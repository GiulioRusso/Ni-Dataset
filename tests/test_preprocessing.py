"""Tests for nidataset.preprocessing on synthetic volumes (FSL-dependent ones marked ``fsl``)."""

import os

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from conftest import save_nifti
from scipy.ndimage import maximum_filter1d

import nidataset as nid

# Fast + stable on the tiny synthetic volumes (see examples/ct_pipeline/pipeline.yaml).
FAST_REG = {"number_iterations": 20, "metric_sampling_percentage": 1.0}


@pytest.mark.parametrize(("view", "axis"), [("axial", 2), ("coronal", 1), ("sagittal", 0)])
def test_mip_is_sliding_window_max(tmp_path, view, axis):
    data = np.random.default_rng(0).random((7, 6, 5)).astype(np.float32)
    src = save_nifti(tmp_path / "v.nii.gz", data)
    out = nid.mip(src, str(tmp_path / "o"), window_size=1, view=view)
    assert os.path.basename(out) == f"v_mip_{view}.nii.gz"
    img = nib.load(out)
    np.testing.assert_allclose(img.get_fdata(), maximum_filter1d(data, size=3, axis=axis, mode="nearest"))
    np.testing.assert_allclose(img.affine, nib.load(src).affine)


@pytest.mark.parametrize(("saving_mode", "rel"), [("case", "case01/axial"), ("view", "axial")])
def test_mip_dataset_layout(ct_dataset, tmp_path, saving_mode, rel):
    paths = nid.mip_dataset(ct_dataset["images"], str(tmp_path / "o"), window_size=2, saving_mode=saving_mode)
    assert len(paths) == 2
    assert os.path.isfile(tmp_path / "o" / rel / "case01_mip_axial.nii.gz")
    with pytest.raises(ValueError):
        nid.mip_dataset(ct_dataset["images"], str(tmp_path / "x"), saving_mode="bogus")


def test_resampling_shape_and_range(ct_dataset, tmp_path):
    src = os.path.join(ct_dataset["images"], "case01.nii.gz")
    out = nid.resampling(src, str(tmp_path / "o"), desired_volume=(20, 20, 16))
    img = nib.load(out)
    assert img.shape == (20, 20, 16)
    before = nib.load(src).get_fdata()
    assert before.min() <= img.get_fdata().min() and img.get_fdata().max() <= before.max()  # clamped
    np.testing.assert_allclose(img.header.get_zooms(), (2.0, 2.0, 3.0), rtol=1e-5)  # same field of view
    with pytest.raises(ValueError, match="three values"):
        nid.resampling(src, str(tmp_path / "o"), desired_volume=(20, 20))


@pytest.mark.parametrize(
    ("saving_mode", "rel"), [("case", "case02/case02_resampled.nii.gz"), ("folder", "case02_resampled.nii.gz")]
)
def test_resampling_dataset_layout(ct_dataset, tmp_path, saving_mode, rel):
    nid.resampling_dataset(ct_dataset["images"], str(tmp_path / "o"), (16, 16, 8), saving_mode=saving_mode)
    assert nib.load(str(tmp_path / "o" / rel)).shape == (16, 16, 8)


def test_registration_chain_case_mode(ct_dataset, tmp_path):
    reg = str(tmp_path / "reg")
    nid.register_CTA_dataset(
        ct_dataset["images"],
        ct_dataset["masks"],
        ct_dataset["template"],
        ct_dataset["template_mask"],
        reg,
        saving_mode="case",
        cleanup=True,
        **FAST_REG,
    )
    template_shape = nib.load(ct_dataset["template"]).shape
    for case in ("case01", "case02"):
        assert nib.load(os.path.join(reg, case, f"{case}_registered.nii.gz")).shape == template_shape
        assert os.path.isfile(os.path.join(reg, case, f"{case}_transformation.tfm"))
        assert not os.path.exists(os.path.join(reg, case, f"{case}_gaussian_filtered.nii.gz"))  # cleanup

    masks = nid.register_mask_dataset(ct_dataset["masks"], reg, reg, str(tmp_path / "m"), saving_mode="case")
    assert len(masks) == 2
    mask = nib.load(masks[0])
    assert mask.shape == template_shape
    assert set(np.unique(mask.get_fdata())) <= {0.0, 1.0}  # nearest neighbour keeps labels

    anns = nid.register_annotation_dataset(
        ct_dataset["annotations"], reg, reg, str(tmp_path / "a"), recalculate_bbox=True, saving_mode="case"
    )
    assert len(anns) == 2
    ann = nib.load(anns[0]).get_fdata()
    assert ann.shape == template_shape
    # recalculated bbox is a solid box
    idx = np.argwhere(ann > 0)
    lo, hi = idx.min(0), idx.max(0)
    assert (ann[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] > 0).all()


def test_register_mask_skips_cases_without_transform(ct_dataset, tmp_path, caplog):
    out = nid.register_mask_dataset(ct_dataset["masks"], str(tmp_path), str(tmp_path), str(tmp_path / "m"))
    assert out == []
    assert "Transformation file not found" in caplog.text


def test_registration_rejects_bad_saving_mode(ct_dataset, tmp_path):
    with pytest.raises(ValueError):
        nid.register_CTA_dataset(
            ct_dataset["images"], ct_dataset["masks"], "t", "tm", str(tmp_path), saving_mode="bogus"
        )
    for func, folder in (
        (nid.register_mask_dataset, ct_dataset["masks"]),
        (nid.register_annotation_dataset, ct_dataset["annotations"]),
    ):
        with pytest.raises(ValueError):
            func(folder, str(tmp_path), str(tmp_path), str(tmp_path / "o"), saving_mode="bogus")


def test_register_cta_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        nid.register_CTA("a.nii.gz", "b.nii.gz", "c.nii.gz", "d.nii.gz", str(tmp_path))


def test_skull_cta_missing_input(ct_dataset, tmp_path):
    with pytest.raises(FileNotFoundError):
        nid.skull_CTA(str(tmp_path / "missing.nii.gz"), str(tmp_path / "o"))
    with pytest.raises(ValueError):
        nid.skull_CTA_dataset(ct_dataset["images"], str(tmp_path / "o"), saving_mode="bogus")


@pytest.mark.fsl
def test_skull_cta_runs_with_fsl(ct_dataset, tmp_path):
    # Folder name has a space: FSL works in a temp dir, so this must still succeed.
    out_dir = tmp_path / "with space"
    paths = nid.skull_CTA_dataset(ct_dataset["images"], str(out_dir), saving_mode="folder", cleanup=True)
    assert len(paths) == 2
    stripped = nib.load(paths[0]).get_fdata()
    assert stripped.min() >= 0 and stripped.max() <= 200  # clip_value default
    assert sorted(os.listdir(out_dir / "masks")) == ["case01_skulled_mask.nii.gz", "case02_skulled_mask.nii.gz"]


def test_save_nifti_helper_roundtrip(tmp_path):
    p = save_nifti(tmp_path / "a" / "b.nii.gz", np.ones((2, 2, 2), np.float32))
    assert nib.load(p).shape == (2, 2, 2)


def _register(ct_dataset, out, **kwargs):
    nid.register_CTA(
        os.path.join(ct_dataset["images"], "case02.nii.gz"),
        os.path.join(ct_dataset["masks"], "case02_mask.nii.gz"),
        ct_dataset["template"],
        ct_dataset["template_mask"],
        str(out),
        **kwargs,
    )
    return sitk.ReadTransform(str(out / "case02_transformation.tfm")).GetParameters()


def test_registration_is_reproducible_with_a_seed(ct_dataset, tmp_path):
    first = _register(ct_dataset, tmp_path / "a", number_iterations=30, metric_sampling_percentage=0.3)
    second = _register(ct_dataset, tmp_path / "b", number_iterations=30, metric_sampling_percentage=0.3)
    np.testing.assert_allclose(first, second, atol=1e-9)  # float noise only
    with pytest.raises(ValueError, match="positive"):
        _register(ct_dataset, tmp_path / "c", random_seed=0)  # SimpleITK's wall-clock sentinel


def test_registration_honours_explicit_learning_rate(ct_dataset, tmp_path):
    # learning_rate=0 -> the optimizer cannot move: rotation stays at the initializer's 0.
    params = _register(ct_dataset, tmp_path / "a", learning_rate=0.0, **FAST_REG)
    assert params[:3] == (0.0, 0.0, 0.0)
    moved = _register(ct_dataset, tmp_path / "b", **FAST_REG)  # estimated rate
    assert moved[:3] != (0.0, 0.0, 0.0)
