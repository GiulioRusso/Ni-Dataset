"""Tests for nidataset.pipeline: validation, ordering, manifest, end-to-end run."""

import json
import os
import shutil

import pytest
from conftest import EXAMPLE_DIR

import nidataset as nid
from nidataset.cli import main as nii_main
from nidataset.pipeline import OPS, PipelineConfig, PipelineConfigError, PipelineError, plan, run_pipeline


def _cfg(tmp_path, steps, **extra):
    data = {"dataset": {"images": str(tmp_path)}, "output": str(tmp_path / "out"), "steps": steps, **extra}
    return PipelineConfig.from_dict(data, base_dir=str(tmp_path))


def _problems(exc_info):
    return "\n".join(exc_info.value.problems)


# Schema / validation


def test_schema_reports_every_structural_problem():
    with pytest.raises(PipelineConfigError) as exc:
        PipelineConfig.from_dict(
            {
                "bogus": 1,
                "dataset": {"view": "x"},
                "steps": [{"op": "mip", "views": ["axial", "oblique"]}, {"op": "mip"}, "nope"],
            }
        )
    msg = _problems(exc)
    for needle in ("bogus", "reserved", "'output'", "oblique", "duplicate step names ['mip']", "step 3"):
        assert needle in msg


def test_paths_resolve_against_config_folder(tmp_path):
    cfg = PipelineConfig.from_dict(
        {"dataset": {"images": "imgs"}, "output": "out", "steps": [{"op": "mip"}]}, base_dir=str(tmp_path)
    )
    assert cfg.dataset["images"] == str(tmp_path / "imgs")
    assert cfg.output == str(tmp_path / "out")


def test_plan_reports_all_problems_before_running(tmp_path):
    cfg = _cfg(
        tmp_path,
        [
            {"op": "no_such_op"},
            {"op": "mip", "nii_folder": "{images}", "bogus_kwarg": 1},
            {"op": "resample", "nii_folder": "{images}"},  # missing desired_volume
            {"op": "extract_slices", "nii_folder": "{nope}"},
            {"op": "qc_volume", "nii_path": str(tmp_path / "missing.nii.gz")},
            {"op": "qc_dataset", "views": ["axial"], "dataset_path": "{images}"},
        ],
    )
    with pytest.raises(PipelineConfigError) as exc:
        plan(cfg)
    msg = _problems(exc)
    assert "unknown op 'no_such_op'" in msg
    assert "bogus_kwarg" in msg
    assert "desired_volume" in msg
    assert "unknown placeholder {nope}" in msg
    assert "missing.nii.gz' does not exist" in msg
    assert "takes no view" in msg
    assert not os.path.exists(cfg.output)  # nothing ran


def test_plan_requires_fsl_for_skull_strip(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    cfg = _cfg(tmp_path, [{"op": "skull_strip", "nii_folder": "{images}"}])
    with pytest.raises(PipelineConfigError, match=r"needs \['fslmaths', 'bet'\].*Install FSL"):
        plan(cfg)


def test_inputs_under_output_may_not_exist_yet(tmp_path):
    cfg = _cfg(tmp_path, [{"op": "qc_volume", "nii_path": "{output}/earlier/scan.nii.gz"}])
    (call,) = plan(cfg)
    assert call.kwargs["nii_path"] == str(tmp_path / "out" / "earlier" / "scan.nii.gz")
    assert call.kwargs["output_path"] == str(tmp_path / "out" / "qc_volume")


# Ordering and manifest (with stub ops)


def test_steps_and_views_run_in_order(tmp_path, monkeypatch):
    calls = []

    def first(nii_folder, output_path, view="axial", size=None):
        calls.append(("first", view, size))
        os.makedirs(output_path, exist_ok=True)
        return 7

    def second(nii_path, output_path):
        calls.append(("second", nii_path, None))

    monkeypatch.setitem(OPS, "first", first)
    monkeypatch.setitem(OPS, "second", second)
    manifest = run_pipeline(
        {
            "dataset": {"images": "."},
            "output": "out",
            "steps": [
                {"op": "first", "views": ["sagittal", "axial"], "nii_folder": "{images}", "size": [2, 3]},
                {"op": "second", "nii_path": "{output}/first/{view}.nii.gz", "views": ["coronal"]},
            ],
        },
        base_dir=str(tmp_path),
    )
    assert calls == [
        ("first", "sagittal", (2, 3)),  # YAML lists arrive as tuples
        ("first", "axial", (2, 3)),
        ("second", str(tmp_path / "out" / "first" / "coronal.nii.gz"), None),
    ]
    assert [(s["step"], s["view"]) for s in manifest["steps"]] == [
        ("first", "sagittal"),
        ("first", "axial"),
        ("second", "coronal"),
    ]

    on_disk = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["status"] == "ok"
    assert manifest["nidataset_version"] == nid.__version__
    assert len(manifest["config_sha256"]) == 64
    step = manifest["steps"][0]
    assert step["result"] == 7
    assert step["status"] == "ok" and step["seconds"] >= 0
    assert step["output_path"] == str(tmp_path / "out" / "first")
    assert step["kwargs"]["size"] == [2, 3]


def test_failed_step_is_recorded_and_stops_the_run(tmp_path, monkeypatch):
    ran = []

    def boom(output_path):
        raise RuntimeError("kaput")

    monkeypatch.setitem(OPS, "boom", boom)
    monkeypatch.setitem(OPS, "after", lambda output_path: ran.append(1))
    with pytest.raises(PipelineError, match="step 'boom' \\(boom\\) failed: kaput"):
        run_pipeline({"output": "out", "steps": [{"op": "boom"}, {"op": "after"}]}, base_dir=str(tmp_path))
    assert ran == []
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert [s["status"] for s in manifest["steps"]] == ["failed"]
    assert "kaput" in manifest["steps"][0]["error"]


def test_config_hash_is_stable_and_content_sensitive(tmp_path, monkeypatch):
    monkeypatch.setitem(OPS, "noop", lambda output_path: None)
    base = {"output": "out", "steps": [{"op": "noop"}]}
    h1 = run_pipeline(base, base_dir=str(tmp_path))["config_sha256"]
    h2 = run_pipeline(dict(reversed(list(base.items()))), base_dir=str(tmp_path))["config_sha256"]
    h3 = run_pipeline({**base, "dataset": {"x": "."}}, base_dir=str(tmp_path))["config_sha256"]
    assert h1 == h2 != h3


# End to end on synthetic data


def test_example_pipeline_end_to_end(ct_dataset):
    root = ct_dataset["root"]
    pytest.importorskip("yaml")
    shutil.copy(os.path.join(EXAMPLE_DIR, "pipeline.yaml"), root)
    manifest = run_pipeline(os.path.join(root, "pipeline.yaml"))

    assert manifest["status"] == "ok"
    assert manifest["config"] == os.path.join(root, "pipeline.yaml")
    out = os.path.join(root, "output")
    for rel in (
        "run_manifest.json",
        "qc_dataset/qc_dataset.json",
        "qc_volume/case01_qc.json",
        "extract_slices/axial/case01_axial_015.tif",
        "extract_masks/coronal_slices_stats.csv",
        "extract_annotations/sagittal/case01_sagittal_015.csv",
        "draw_annotations/case01_axial_015_annotated.png",
        "register/transforms/case01_transformation.tfm",
        "register_mask/case02_registered_mask.nii.gz",
        "register_annotation/case01_registered_bbox.nii.gz",
        "resample/case01_resampled.nii.gz",
    ):
        assert os.path.exists(os.path.join(out, rel)), rel
    assert {s["op"] for s in manifest["steps"]} >= {"register", "mip", "qc_dataset"}


def test_nii_run_cli_with_json_config(ct_dataset, capsys):
    cfg = {
        "dataset": {"images": "data/images"},
        "output": "out",
        "steps": [{"op": "mip", "nii_folder": "{images}", "window_size": 3, "saving_mode": "view"}],
    }
    path = os.path.join(ct_dataset["root"], "pipeline.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    assert nii_main(["run", path]) == 0
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert printed == os.path.join(ct_dataset["root"], "out", "run_manifest.json")


def test_nii_run_cli_invalid_config_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"output": "out", "steps": [{"op": "nope"}]}))
    assert nii_main(["run", str(path)]) == 2
    assert "unknown op 'nope'" in capsys.readouterr().err


def test_only_runs_selected_steps_in_config_order(tmp_path, monkeypatch, capsys):
    ran = []
    for name in ("a", "b", "c"):
        monkeypatch.setitem(OPS, name, lambda output_path, _n=name: ran.append(_n))
    cfg = {"output": "out", "steps": [{"op": "a"}, {"op": "b"}, {"op": "c"}]}
    manifest = run_pipeline(cfg, base_dir=str(tmp_path), only=["c", "a"])
    assert ran == ["a", "c"]
    assert manifest["only"] == ["c", "a"]
    with pytest.raises(PipelineConfigError, match="unknown step names \\['nope'\\]"):
        run_pipeline(cfg, base_dir=str(tmp_path), only=["nope"])

    path = tmp_path / "p.json"
    path.write_text(json.dumps(cfg))
    ran.clear()
    assert nii_main(["run", str(path), "--only", "b"]) == 0
    assert ran == ["b"]
