"""Behavioural tests for analysis, utility, draw, transforms and volume helpers."""

import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from conftest import save_nifti

import nidataset as nid


@pytest.fixture
def pair(tmp_path):
    a = np.zeros((10, 10, 10), np.float32)
    a[2:6, 2:6, 2:6] = 1
    b = np.zeros_like(a)
    b[3:7, 2:6, 2:6] = 1
    return save_nifti(tmp_path / "a" / "x.nii.gz", a), save_nifti(tmp_path / "b" / "x.nii.gz", b)


def _data(path):
    return nib.load(path).get_fdata()


# analysis


def test_compare_volumes_metrics(pair):
    a, b = pair
    same = nid.compare_volumes(a, a)
    assert same["dice"] == 1.0 and same["mse"] == 0.0
    res = nid.compare_volumes(a, b, metrics=["dice", "jaccard"])
    assert set(res) == {"dice", "jaccard"}
    assert res["dice"] == pytest.approx(2 * 48 / 128)  # overlap 3x4x4 of two 4x4x4 boxes
    assert res["jaccard"] == pytest.approx(48 / 80)


def test_compare_volumes_dataset_csv(pair, tmp_path):
    a, b = pair
    out = nid.compare_volumes_dataset(os.path.dirname(a), os.path.dirname(b), str(tmp_path / "o"))
    df = pd.read_csv(out)
    assert len(df) == 1 and df["DICE"].iloc[0] == pytest.approx(0.75)


def test_volume_statistics(pair, tmp_path):
    a, _ = pair
    stats = nid.compute_volume_statistics(a)
    assert stats["max"] == 1.0 and stats["min"] == 0.0
    assert stats["mean"] == pytest.approx(64 / 1000)
    masked = nid.compute_volume_statistics(a, mask_path=a)
    assert masked["mean"] == 1.0 and masked["std"] == 0.0
    out = nid.compute_volume_statistics_dataset(os.path.dirname(a), str(tmp_path / "o"))
    assert len(pd.read_csv(out)) == 1


def test_split_dataset_is_seeded_and_complete(ct_dataset, tmp_path):
    for i in range(3, 6):
        save_nifti(os.path.join(ct_dataset["images"], f"case0{i}.nii.gz"), np.zeros((2, 2, 2), np.float32))
    s1 = nid.split_dataset(ct_dataset["images"], str(tmp_path / "s1"), ratios=(0.6, 0.2, 0.2), seed=1)
    s2 = nid.split_dataset(ct_dataset["images"], str(tmp_path / "s2"), ratios=(0.6, 0.2, 0.2), seed=1)
    assert s1 == s2
    assert sorted(f for files in s1.values() for f in files) == sorted(os.listdir(ct_dataset["images"]))
    assert [len(s1[k]) for k in ("train", "val", "test")] == [3, 1, 1]
    with pytest.raises(ValueError):
        nid.split_dataset(ct_dataset["images"], str(tmp_path / "bad"), ratios=(0.5, 0.2, 0.2))


# utility


def test_dataset_info_tables(ct_dataset, tmp_path):
    rows = nid.dataset_images_info(ct_dataset["images"], str(tmp_path / "i"))
    assert [r[0] for r in rows] == ["case01.nii.gz", "case02.nii.gz"]
    assert tuple(rows[0][1]) == (40, 40, 32)
    anns = nid.dataset_annotations_info(ct_dataset["annotations"], str(tmp_path / "a"))
    assert [int(v) for v in anns[0][1][0]] == [12, 12, 12, 19, 19, 19]
    assert any(f.endswith(".csv") for f in os.listdir(tmp_path / "i"))


# draw


def test_draw_3d_boxes_and_2d_coords(ct_dataset, tmp_path):
    src = os.path.join(ct_dataset["images"], "case01.nii.gz")
    df = pd.DataFrame(
        {"SCORE": [0.9], "X_MIN": [5], "Y_MIN": [6], "Z_MIN": [7], "X_MAX": [9], "Y_MAX": [10], "Z_MAX": [11]}
    )
    out = nid.draw_3D_boxes(df, src, str(tmp_path / "o"), intensity_based_on_score=True)
    drawn = _data(out)
    assert drawn.shape == (40, 40, 32)
    nz = np.argwhere(drawn > 0)
    assert tuple(nz.min(0)) == (5, 6, 7) and tuple(nz.max(0)) == (9, 10, 11)  # MAX inclusive
    assert set(np.unique(drawn)) == {0.0, 3.0}  # score 0.9 -> top intensity level

    # Boxes poking outside the volume are clipped, never wrapped around.
    edge = pd.DataFrame({"X_MIN": [-3], "Y_MIN": [0], "Z_MIN": [0], "X_MAX": [1], "Y_MAX": [1], "Z_MAX": [1]})
    clipped = _data(nid.draw_3D_boxes(edge, src, str(tmp_path / "e")))
    assert clipped.sum() == 2 * 2 * 2 and clipped[-1].sum() == 0

    df2d = pd.DataFrame({"X": [1], "Y": [2], "SLICE_NUMBER": [3]})
    assert nid.from_2D_to_3D_coords(df2d, "axial").iloc[0].tolist() == [2, 1, 3]
    with pytest.raises(ValueError):
        nid.from_2D_to_3D_coords(df2d, "oblique")


@pytest.mark.parametrize(
    "columns",
    [{"CENTER_X": [5], "CENTER_Y": [5]}, {"X_MIN": [2], "Y_MIN": [2], "X_MAX": [8], "Y_MAX": [8]}],
)
def test_draw_2d_annotations_formats(tmp_path, columns):
    import cv2

    img = tmp_path / "s.png"
    cv2.imwrite(str(img), np.zeros((12, 12), np.uint8))
    csv = tmp_path / "s.csv"
    pd.DataFrame(columns).to_csv(csv, index=False)
    out = nid.draw_2D_annotations(str(csv), str(img), str(tmp_path / "o"), radius=2)
    drawn = cv2.imread(out)
    assert drawn.shape == (12, 12, 3) and drawn.any()

    pd.DataFrame({"CENTER_X": [1], "CENTER_Y": [1], "CENTER_Z": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="slice-format"):
        nid.draw_2D_annotations(str(csv), str(img), str(tmp_path / "o"))


# transforms


@pytest.mark.parametrize("method", ["zscore", "minmax", "percentile", "histogram"])
def test_intensity_normalization(ct_dataset, tmp_path, method):
    src = os.path.join(ct_dataset["images"], "case01.nii.gz")
    ref = os.path.join(ct_dataset["images"], "case02.nii.gz")
    data = _data(nid.intensity_normalization(src, str(tmp_path / "o"), method=method, reference_path=ref))
    if method == "zscore":
        assert abs(data.mean()) < 1e-3 and data.std() == pytest.approx(1.0, abs=1e-3)
    elif method in ("minmax", "percentile"):
        assert data.min() >= 0.0 and data.max() <= 1.0 + 1e-6
    assert data.shape == (40, 40, 32)


def test_windowing(ct_dataset, tmp_path):
    src = os.path.join(ct_dataset["images"], "case01.nii.gz")
    norm = _data(nid.windowing(src, str(tmp_path / "a"), preset="brain"))
    assert norm.min() >= 0 and norm.max() <= 1
    raw = _data(nid.windowing(src, str(tmp_path / "b"), window_center=40, window_width=20, normalize=False))
    assert raw.min() >= 30 and raw.max() <= 50
    with pytest.raises(ValueError):
        nid.windowing(src, str(tmp_path / "c"), preset="nope")
    assert len(nid.windowing_dataset(ct_dataset["images"], str(tmp_path / "d"), preset="stroke")) == 2


def test_resample_to_reference_and_apply_transform(ct_dataset, tmp_path):
    src = os.path.join(ct_dataset["images"], "case01.nii.gz")
    ref = save_nifti(tmp_path / "ref.nii.gz", np.zeros((20, 20, 16), np.float32), np.diag([2, 2, 3, 1.0]))
    out = nid.resample_to_reference(src, ref, str(tmp_path / "r"), interpolation="nearest")
    assert nib.load(out).shape == (20, 20, 16)
    assert len(nid.resample_to_reference_dataset(ct_dataset["images"], ref, str(tmp_path / "rd"))) == 2

    tfm = str(tmp_path / "identity.tfm")
    sitk.WriteTransform(sitk.Euler3DTransform(), tfm)
    moved = nid.apply_transform(src, tfm, src, str(tmp_path / "t"))
    np.testing.assert_allclose(_data(moved), _data(src), atol=1e-4)


@pytest.mark.parametrize("compressed", [True, False])
def test_numpy_roundtrip(ct_dataset, tmp_path, compressed):
    src = os.path.join(ct_dataset["images"], "case01.nii.gz")
    arr = nid.nifti_to_numpy(src, str(tmp_path / "np"), compressed=compressed)
    back = nid.numpy_to_nifti(arr, str(tmp_path / "back"), reference_nifti=src)
    np.testing.assert_allclose(_data(back), _data(src), atol=1e-5)
    np.testing.assert_allclose(nib.load(back).affine, nib.load(src).affine)


def test_rescale_dataset(ct_dataset, tmp_path):
    outs = nid.rescale_intensity_dataset(ct_dataset["images"], str(tmp_path / "o"), out_min=-1, out_max=1)
    assert [round(float(_data(p).max()), 4) for p in outs] == [1.0, 1.0]


# volume


def test_bounding_boxes_from_mask(ct_dataset, tmp_path):
    ann = os.path.join(ct_dataset["annotations"], "case01_bbox.nii.gz")
    out = nid.extract_bounding_boxes(ann, str(tmp_path / "o"), voxel_size=(1, 1, 1), volume_threshold=10)
    nz = np.argwhere(_data(out) > 0)
    assert tuple(nz.min(0)) == (12, 12, 12) and tuple(nz.max(0)) == (19, 19, 19)
    outs = nid.extract_bounding_boxes_dataset(
        ct_dataset["annotations"], str(tmp_path / "d"), voxel_size=(1, 1, 1), volume_threshold=10
    )
    assert len(outs) == 2


def test_brain_mask_and_heatmap(ct_dataset, tmp_path):
    src = os.path.join(ct_dataset["images"], "case01.nii.gz")
    mask = _data(nid.generate_brain_mask(src, str(tmp_path / "m"), threshold=(20, 100), closing_radius=1))
    truth = _data(os.path.join(ct_dataset["masks"], "case01_mask.nii.gz"))
    assert (
        nid.compare_volumes(
            *(save_nifti(tmp_path / f"{n}.nii.gz", v) for n, v in (("m", mask), ("t", truth))), metrics=["dice"]
        )["dice"]
        > 0.99
    )  # recovers the synthetic head

    # Otsu path (threshold=None) on a noise-free volume.
    clean = np.where(truth > 0, 40.0, 0.0).astype(np.float32)
    clean[15:20, 15:20, 15:20] = 70.0
    auto = _data(nid.generate_brain_mask(save_nifti(tmp_path / "clean.nii.gz", clean), str(tmp_path / "a")))
    assert (auto == truth).mean() > 0.99
    assert len(nid.generate_brain_mask_dataset(ct_dataset["images"], str(tmp_path / "md"), threshold=(20, 100))) == 2

    heat = _data(nid.generate_heatmap_volume(ct_dataset["annotations"], str(tmp_path / "h")))
    assert heat.shape == (40, 40, 32) and heat.max() > 0


def test_swap_views_and_crop_pad_dataset(ct_dataset, tmp_path):
    src = os.path.join(ct_dataset["images"], "case01.nii.gz")
    swapped = nib.load(nid.swap_nifti_views(src, str(tmp_path / "s"), "axial", "coronal"))
    assert sorted(swapped.shape) == sorted((40, 40, 32))
    outs = nid.crop_and_pad_dataset(ct_dataset["images"], str(tmp_path / "c"), target_shape=(36, 36, 30))
    assert all(nib.load(p).shape == (36, 36, 30) for p in outs)


def test_extracted_volume_boxes_draw_back_onto_the_annotation(ct_dataset, tmp_path):
    # extract_annotations (volume/box) -> draw_3D_boxes must reproduce the original box.
    ann = os.path.join(ct_dataset["annotations"], "case01_bbox.nii.gz")
    (csv,) = nid.extract_annotations(ann, str(tmp_path / "csv"), saving_mode="volume", data_mode="box")
    boxes = pd.read_csv(csv)
    # axial box CSV stores 2D X = Y_3D, 2D Y = X_3D; map back to volume axes
    boxes = boxes.rename(columns={"X_MIN": "Y_MIN", "Y_MIN": "X_MIN", "X_MAX": "Y_MAX", "Y_MAX": "X_MAX"})
    drawn = _data(nid.draw_3D_boxes(boxes, ann, str(tmp_path / "d")))
    np.testing.assert_array_equal(drawn, _data(ann))
