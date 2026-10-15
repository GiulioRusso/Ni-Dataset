"""Tests for nidataset.slices: slice and annotation extraction, single file and dataset."""

import csv
import os

import numpy as np
import pytest
from conftest import save_nifti
from PIL import Image

import nidataset as nid

SHAPE = (6, 5, 4)


@pytest.fixture
def volume(tmp_path):
    return save_nifti(tmp_path / "vol.nii.gz", np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE))


@pytest.fixture
def annotation(tmp_path):
    # Box label 1 spanning x 1..3, y 1..2, z 2..3 (inclusive).
    data = np.zeros(SHAPE, dtype=np.uint8)
    data[1:4, 1:3, 2:4] = 1
    return save_nifti(tmp_path / "ann_bbox.nii.gz", data)


def _rows(path):
    with open(path) as f:
        return list(csv.reader(f))


@pytest.mark.parametrize(
    ("view", "count", "hw"), [("axial", 4, (6, 5)), ("coronal", 5, (6, 4)), ("sagittal", 6, (5, 4))]
)
def test_extract_slices_count_names_and_content(tmp_path, volume, view, count, hw):
    paths = nid.extract_slices(volume, str(tmp_path / "out"), view=view)
    assert len(paths) == count
    assert os.path.basename(paths[1]) == f"vol_{view}_001.tif"
    sl = np.array(Image.open(paths[0]))
    assert sl.shape == hw
    expected = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    index = {"axial": (slice(None), slice(None), 0), "coronal": (slice(None), 0), "sagittal": (0,)}[view]
    assert np.array_equal(sl, expected[index])


def test_extract_slices_padding(tmp_path, volume):
    paths = nid.extract_slices(volume, str(tmp_path / "out"), target_size=(9, 8), pad_value=-1)
    sl = np.array(Image.open(paths[0]))
    assert sl.shape == (9, 8)
    assert sl[0, 0] == -1 and sl[-1, -1] == -1  # odd padding goes right/bottom
    with pytest.raises(ValueError, match="smaller than slice"):
        nid.extract_slices(volume, str(tmp_path / "o2"), target_size=(2, 2))


def test_extract_slices_rejects_bad_input(tmp_path, volume):
    with pytest.raises(ValueError):
        nid.extract_slices(volume, str(tmp_path / "o"), view="oblique")
    with pytest.raises(FileNotFoundError):
        nid.extract_slices(str(tmp_path / "nope.nii.gz"), str(tmp_path / "o"))
    four_d = save_nifti(tmp_path / "4d.nii.gz", np.zeros((2, 2, 2, 2), np.float32))
    with pytest.raises(ValueError, match="3D"):
        nid.extract_slices(four_d, str(tmp_path / "o"))


@pytest.mark.parametrize("saving_mode", ["case", "view"])
def test_extract_slices_dataset_layout_and_stats(tmp_path, volume, saving_mode):
    total = nid.extract_slices_dataset(
        str(tmp_path), str(tmp_path / "out"), view="coronal", saving_mode=saving_mode, save_stats=True
    )
    assert total == 5
    folder = tmp_path / "out" / ("vol/coronal" if saving_mode == "case" else "coronal")
    assert len(os.listdir(folder)) == 5
    rows = _rows(tmp_path / "out" / "coronal_slices_stats.csv")
    assert rows[0] == ["FILENAME", "NUM_SLICES"] and rows[-1] == ["TOTAL_SLICES", "5"]
    with pytest.raises(ValueError):
        nid.extract_slices_dataset(str(tmp_path), str(tmp_path / "x"), saving_mode="bogus")


def test_extract_annotations_center_slice_mode(tmp_path, annotation):
    paths = nid.extract_annotations(annotation, str(tmp_path / "out"), view="axial", data_mode="center")
    # bbox spans z 2..3 -> one CSV per slice, prefix drops "_bbox"
    assert sorted(os.path.basename(p) for p in paths) == ["ann_axial_002.csv", "ann_axial_003.csv"]
    header, row = _rows(paths[0])
    assert header == ["CENTER_X", "CENTER_Y"]
    # axial: 2D X = Y_3D center (1.5), 2D Y = X_3D center (2.0)
    assert [float(v) for v in row] == [1.5, 2.0]


@pytest.mark.parametrize(
    ("view", "data_mode", "header", "row", "first_slice"),
    [
        ("coronal", "box", ["X_MIN", "Y_MIN", "X_MAX", "Y_MAX"], [2, 1, 3, 3], 1),
        ("sagittal", "box", ["X_MIN", "Y_MIN", "X_MAX", "Y_MAX"], [2, 1, 3, 2], 1),
        ("axial", "radius", ["CENTER_X", "CENTER_Y", "RADIUS_X", "RADIUS_Y"], [1.5, 2.0, 0.5, 1.0], 2),
        ("coronal", "radius", ["CENTER_X", "CENTER_Y", "RADIUS_X", "RADIUS_Y"], [2.5, 2.0, 0.5, 1.0], 1),
        ("sagittal", "radius", ["CENTER_X", "CENTER_Y", "RADIUS_X", "RADIUS_Y"], [2.5, 1.5, 0.5, 0.5], 1),
        ("sagittal", "center", ["CENTER_X", "CENTER_Y"], [2.5, 1.5], 1),
    ],
)
def test_extract_annotations_modes(tmp_path, annotation, view, data_mode, header, row, first_slice):
    paths = sorted(nid.extract_annotations(annotation, str(tmp_path / "o"), view=view, data_mode=data_mode))
    assert paths[0].endswith(f"_{first_slice:03d}.csv")
    got = _rows(paths[0])
    assert got[0] == header
    assert [float(v) for v in got[1]] == row


@pytest.mark.parametrize(
    ("data_mode", "header"),
    [
        ("center", ["CENTER_X", "CENTER_Y", "CENTER_Z"]),
        ("box", ["X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"]),
        ("radius", ["CENTER_X", "CENTER_Y", "CENTER_Z", "RADIUS_X", "RADIUS_Y", "RADIUS_Z"]),
    ],
)
def test_extract_annotations_volume_mode(tmp_path, annotation, data_mode, header):
    (path,) = nid.extract_annotations(annotation, str(tmp_path / "o"), saving_mode="volume", data_mode=data_mode)
    assert os.path.basename(path) == "ann.csv"
    rows = _rows(path)
    assert rows[0] == header and len(rows) == 2


def test_extract_annotations_padding_offsets_match_slices(tmp_path, annotation):
    # axial slice is (6, 5); padding to (10, 9) shifts rows by 2 and columns by 2.
    (path, _) = sorted(
        nid.extract_annotations(annotation, str(tmp_path / "o"), data_mode="center", target_size=(10, 9))
    )
    assert [float(v) for v in _rows(path)[1]] == [1.5 + 2, 2.0 + 2]
    with pytest.raises(ValueError, match="smaller than slice"):
        nid.extract_annotations(annotation, str(tmp_path / "o2"), target_size=(2, 2))


def test_extract_annotations_rejects_bad_modes(tmp_path, annotation):
    with pytest.raises(ValueError):
        nid.extract_annotations(annotation, str(tmp_path / "o"), saving_mode="bogus")
    with pytest.raises(ValueError):
        nid.extract_annotations(annotation, str(tmp_path / "o"), data_mode="bogus")


@pytest.mark.parametrize("saving_mode", ["case", "view"])
def test_extract_annotations_dataset(tmp_path, annotation, saving_mode):
    total = nid.extract_annotations_dataset(
        str(tmp_path), str(tmp_path / "out"), view="axial", saving_mode=saving_mode, save_stats=True
    )
    assert total == 1
    folder = tmp_path / "out" / ("ann_bbox/axial" if saving_mode == "case" else "axial")
    assert len(os.listdir(folder)) == 2
    assert _rows(tmp_path / "out" / "axial_annotations_stats.csv")[-1] == ["TOTAL_ANNOTATIONS", "1"]
    for bad in ({"saving_mode": "x"}, {"extraction_mode": "x"}, {"data_mode": "x"}):
        with pytest.raises(ValueError):
            nid.extract_annotations_dataset(str(tmp_path), str(tmp_path / "bad"), **bad)
