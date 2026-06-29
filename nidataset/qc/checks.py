"""
Single-volume QC checks: geometry, data and file integrity.

The geometry checks are the core value of the module. A volume can load fine in
any viewer yet be silently wrong: an unexpected orientation (LAS vs RAS), a
singular affine, anisotropic spacing, or sform/qform that disagree. These do not
crash anything; they quietly corrupt training. Each check below is deliberately
explicit about the geometric convention it assumes.

Axis convention
---------------
All geometry here is expressed in **nibabel** terms: voxel indices ``(i, j, k)``
and a 4x4 ``affine`` mapping voxel -> world (RAS+) coordinates. Orientation codes
come from :func:`nibabel.aff2axcodes`, which reports the world axis each voxel
axis points to (e.g. ``('R', 'A', 'S')``). SimpleITK uses the reversed axis order
``(z, y, x)``; this module never mixes the two — it stays in nibabel space.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import nibabel as nib

from .._helpers import validate_nifti_path, is_nifti, logger
from .config import QCConfig
from .report import Report, OK, WARNING, ERROR

# dtypes that several NIfTI readers handle poorly (the NIfTI-1 spec has no
# official 64-bit integer datatype; many tools truncate or refuse them).
_PROBLEM_INT_DTYPES = ("int64", "uint64")


def _direction(affine: np.ndarray) -> np.ndarray:
    """Return the 3x3 direction/scaling part of a 4x4 affine."""
    return np.asarray(affine, dtype=np.float64)[:3, :3]


def _spatial_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return the first three (spatial) dimensions of a volume shape."""
    return tuple(shape[:3])


# Geometry

def _check_geometry(img: "nib.Nifti1Image", config: QCConfig, report: Report) -> None:
    """Run all affine / orientation / spacing checks on *img* into *report*."""
    affine = img.affine

    # --- affine present & finite ------------------------------------------
    if affine is None:
        report.add("affine_present", ERROR, "Affine is missing.", None)
        return
    if not np.all(np.isfinite(affine)):
        report.add("affine_finite", ERROR, "Affine contains NaN or inf.", None)
        return
    report.add("affine_finite", OK, "Affine is present and finite.", None)

    # --- non-singular ------------------------------------------------------
    # A singular direction matrix (det == 0) collapses a spatial dimension:
    # voxel->world is not invertible, so resampling/registration is meaningless.
    det = float(np.linalg.det(_direction(affine)))
    if abs(det) <= config.affine_atol:
        report.add("affine_nonsingular", ERROR,
                   f"Affine direction matrix is singular (det={det:.3e}).", det)
    else:
        report.add("affine_nonsingular", OK, f"Affine is non-singular (det={det:.3e}).", det)

    # --- orientation -------------------------------------------------------
    axcodes = "".join(nib.aff2axcodes(affine))
    report.meta["orientation"] = axcodes
    if config.expected_orientation is None:
        report.add("orientation", OK, f"Orientation is {axcodes} (no expectation set).", axcodes)
    elif axcodes == config.expected_orientation:
        report.add("orientation", OK, f"Orientation {axcodes} matches expectation.", axcodes)
    else:
        report.add("orientation", ERROR,
                   f"Orientation {axcodes} differs from expected {config.expected_orientation}.",
                   axcodes)

    # --- spacing -----------------------------------------------------------
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    report.meta["spacing"] = zooms
    if not all(np.isfinite(zooms)) or any(z <= 0 for z in zooms):
        report.add("spacing_positive", ERROR, f"Non-positive or non-finite spacing {zooms}.", zooms)
    else:
        report.add("spacing_positive", OK, f"Spacing is positive and finite {zooms}.", zooms)

        # isotropy as max/min - 1
        anisotropy = max(zooms) / min(zooms) - 1.0
        if anisotropy > config.isotropy_tol:
            report.add("spacing_isotropy", WARNING,
                       f"Anisotropic spacing (max/min-1={anisotropy:.3f} > {config.isotropy_tol}).",
                       anisotropy)
        else:
            report.add("spacing_isotropy", OK, f"Spacing isotropic within tol ({anisotropy:.3f}).",
                       anisotropy)

        # plausible range
        lo, hi = config.spacing_range
        if any(z < lo or z > hi for z in zooms):
            report.add("spacing_range", WARNING,
                       f"Spacing {zooms} outside plausible range [{lo}, {hi}].", zooms)
        else:
            report.add("spacing_range", OK, f"Spacing within range [{lo}, {hi}].", zooms)

    # --- sform / qform agreement ------------------------------------------
    # When both codes are set, the two stored affines should describe the same
    # world mapping; divergence means tools that prefer one over the other will
    # place the volume differently in space.
    sform, scode = img.header.get_sform(coded=True)
    qform, qcode = img.header.get_qform(coded=True)
    if scode and qcode:
        max_diff = float(np.max(np.abs(np.asarray(sform) - np.asarray(qform))))
        if max_diff > config.affine_atol:
            report.add("sform_qform_agree", WARNING,
                       f"sform and qform disagree (max|Δ|={max_diff:.3e} > {config.affine_atol}).",
                       max_diff)
        else:
            report.add("sform_qform_agree", OK, f"sform and qform agree (max|Δ|={max_diff:.3e}).",
                       max_diff)


# Data

def _empty_slice_counts(data: np.ndarray, config: QCConfig) -> dict:
    """
    Count "empty" slices along each of the first three axes.

    A slice is empty when its foreground fraction (voxels strictly above
    ``empty_slice_bg_value``) is below ``empty_slice_min_fg_fraction``.
    """
    counts = {}
    for axis in range(min(3, data.ndim)):
        n = data.shape[axis]
        empty = 0
        for idx in range(n):
            sl = np.take(data, idx, axis=axis)
            fg = np.count_nonzero(sl > config.empty_slice_bg_value)
            if sl.size and (fg / sl.size) < config.empty_slice_min_fg_fraction:
                empty += 1
        counts[axis] = (empty, n)
    return counts


def _check_data(img: "nib.Nifti1Image", config: QCConfig, report: Report) -> None:
    """Run NaN/inf, dtype, constancy, intensity-range and empty-slice checks."""
    data = np.asarray(img.get_fdata(dtype=np.float64))

    # --- NaN / inf ---------------------------------------------------------
    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())
    if n_nan or n_inf:
        report.add("finite_values", ERROR, f"Data has {n_nan} NaN and {n_inf} inf voxels.",
                   {"nan": n_nan, "inf": n_inf})
        finite = data[np.isfinite(data)]
    else:
        report.add("finite_values", OK, "No NaN or inf in data.", {"nan": 0, "inf": 0})
        finite = data

    # --- dtype -------------------------------------------------------------
    dtype = str(img.header.get_data_dtype())
    report.meta["dtype"] = dtype
    if dtype in _PROBLEM_INT_DTYPES:
        report.add("dtype", WARNING,
                   f"dtype {dtype} is not portable across NIfTI tools (no 64-bit int in NIfTI-1).",
                   dtype)
    elif dtype == "float64" and config.warn_float64:
        report.add("dtype", WARNING, "dtype float64; float32 usually suffices and halves size.", dtype)
    else:
        report.add("dtype", OK, f"dtype {dtype}.", dtype)

    # --- constant / all-zero ----------------------------------------------
    if finite.size == 0:
        report.add("constant_volume", ERROR, "Volume has no finite voxels.", None)
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            label = "all-zero" if vmin == 0 else f"constant ({vmin})"
            report.add("constant_volume", ERROR, f"Volume is {label}.", vmin)
        else:
            report.add("constant_volume", OK, "Volume is non-constant.", None)

        # --- intensity range -----------------------------------------------
        pcts = np.percentile(finite, [1, 50, 99])
        report.meta["intensity"] = {
            "min": vmin, "max": vmax,
            "p1": float(pcts[0]), "p50": float(pcts[1]), "p99": float(pcts[2]),
        }
        if config.intensity_range is not None:
            lo, hi = config.intensity_range
            if vmin < lo or vmax > hi:
                report.add("intensity_range", WARNING,
                           f"Intensity [{vmin:.3g}, {vmax:.3g}] outside expected [{lo}, {hi}].",
                           [vmin, vmax])
            else:
                report.add("intensity_range", OK, f"Intensity within [{lo}, {hi}].", [vmin, vmax])

    # --- empty slices ------------------------------------------------------
    empty = _empty_slice_counts(data, config)
    worst_axis = None
    worst_frac = 0.0
    for axis, (n_empty, n) in empty.items():
        frac = n_empty / n if n else 0.0
        if frac > worst_frac:
            worst_frac, worst_axis = frac, axis
    report.meta["empty_slices"] = {a: e for a, (e, _) in empty.items()}
    if worst_axis is not None and worst_frac > config.max_empty_slice_fraction:
        report.add("empty_slices", WARNING,
                   f"Axis {worst_axis}: {worst_frac:.0%} of slices empty "
                   f"(> {config.max_empty_slice_fraction:.0%}).", worst_frac)
    else:
        report.add("empty_slices", OK, "Empty-slice fraction within limit.", worst_frac)


# File / shape

def _check_file(path: str, img: "nib.Nifti1Image", report: Report) -> None:
    """Check extension consistency and report shape / dimensionality."""
    if is_nifti(path):
        report.add("extension", OK, f"Valid NIfTI extension ({os.path.basename(path)}).", None)
    else:
        report.add("extension", WARNING, f"Unexpected extension for NIfTI content: {path}.", None)

    shape = tuple(int(s) for s in img.shape)
    report.meta["shape"] = shape
    report.meta["ndim"] = len(shape)
    if len(shape) == 3:
        report.add("dimensionality", OK, f"3D volume {shape}.", shape)
    elif len(shape) == 4:
        # Explicitly surfaced rather than silently assuming 3D; data checks run
        # on the full array, geometry uses the spatial 3 dims.
        report.add("dimensionality", WARNING,
                   f"4D volume {shape} (timeseries/multi-channel); checks treat first 3 dims as spatial.",
                   shape)
    else:
        report.add("dimensionality", ERROR, f"Unexpected dimensionality {len(shape)}D {shape}.", shape)


# Public API

def check_volume(nii_path: str, config: Optional[QCConfig] = None) -> Report:
    """
    Run all single-volume QC checks on a NIfTI file.

    Checks geometry (affine present/finite/non-singular, orientation, spacing,
    sform/qform agreement), data (NaN/inf, dtype portability, constancy,
    intensity range, empty slices) and file integrity (extension, shape, 3D/4D).

    :param nii_path: Path to the ``.nii`` / ``.nii.gz`` file.
    :param config:   Thresholds; defaults to :class:`QCConfig` if ``None``.

    :returns: A :class:`Report` (``kind="volume"``). ``report.status`` is the
        worst of all checks; ``report.meta`` carries shape, dtype, orientation,
        spacing and intensity context.

    Example
    -------
    >>> from nidataset.qc import check_volume, QCConfig
    >>> rep = check_volume("scan.nii.gz", QCConfig(expected_orientation="RAS"))
    >>> rep.status, [r.name for r in rep.issues()]
    """
    config = config or QCConfig()
    report = Report(target=nii_path, kind="volume")

    validate_nifti_path(nii_path)
    try:
        img = nib.load(nii_path)
    except Exception as exc:  # unreadable / corrupt header
        report.add("readable", ERROR, f"Failed to load NIfTI: {exc}", str(exc))
        return report
    report.add("readable", OK, "File loaded successfully.", None)

    _check_file(nii_path, img, report)
    _check_geometry(img, config, report)
    _check_data(img, config, report)

    logger.debug("check_volume(%s) -> %s", nii_path, report.status)
    return report
