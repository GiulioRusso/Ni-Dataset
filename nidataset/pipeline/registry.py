"""
Pipeline op registry: step ``op`` name -> existing nidataset function.

Step kwargs are passed straight to the function, so the function's own
signature and docs are the reference for each op. The runner injects
``output_path``.
"""

from __future__ import annotations

import os

from .._helpers import ensure_dir, strip_nifti_ext
from ..draw import draw_2D_annotations
from ..preprocessing import (
    mip_dataset,
    register_annotation_dataset,
    register_CTA_dataset,
    register_mask_dataset,
    resampling_dataset,
    skull_CTA_dataset,
)
from ..qc import QCConfig, check_dataset, check_volume, to_json
from ..slices import extract_annotations_dataset, extract_slices_dataset


def qc_volume(nii_path: str, output_path: str, config_path: str | None = None) -> dict:
    """Run :func:`nidataset.qc.check_volume` and write ``<PREFIX>_qc.json`` to *output_path*."""
    report = check_volume(nii_path, QCConfig.load(config_path) if config_path else None)
    ensure_dir(output_path)
    out = os.path.join(output_path, f"{strip_nifti_ext(os.path.basename(nii_path))}_qc.json")
    to_json(report, out)
    return {"report": out, "status": report.status, "counts": report.counts()}


def qc_dataset(dataset_path: str, output_path: str, config_path: str | None = None) -> dict:
    """Run :func:`nidataset.qc.check_dataset` (folder or CSV manifest) and write ``qc_dataset.json``."""
    report = check_dataset(dataset_path, QCConfig.load(config_path) if config_path else None)
    ensure_dir(output_path)
    out = os.path.join(output_path, "qc_dataset.json")
    to_json(report, out)
    return {"report": out, "status": report.status, "counts": report.counts()}


OPS = {
    "extract_slices": extract_slices_dataset,
    "extract_annotations": extract_annotations_dataset,
    "draw_annotations": draw_2D_annotations,
    "skull_strip": skull_CTA_dataset,
    "register": register_CTA_dataset,
    "register_mask": register_mask_dataset,
    "register_annotation": register_annotation_dataset,
    "mip": mip_dataset,
    "resample": resampling_dataset,
    "qc_volume": qc_volume,
    "qc_dataset": qc_dataset,
}

# External executables an op shells out to; checked before anything runs.
REQUIRES = {"skull_strip": ("fslmaths", "bet")}
