"""
Pair / triple coherence checks: image <-> mask <-> annotation.

This is the heart of the module for detection / segmentation. Matching *shape*
is not enough: an image and a mask can have identical shapes yet different
affines, which places them at different world positions — the model trains on a
mask that is silently shifted relative to the image. The affine-agreement check
is the one that saves experiments.

``check_pair(image, mask)`` is the two-volume case and ``check_triple(image,
mask, annotation)`` the three-volume case; both share the same geometry-match
core (:func:`_check_geom_match`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import nibabel as nib
from scipy import ndimage

from .._helpers import validate_nifti_path, logger
from .checks import _spatial_shape
from .config import QCConfig
from .report import Report, OK, WARNING, ERROR


def _load(path: str) -> "nib.Nifti1Image":
    validate_nifti_path(path)
    return nib.load(path)


def _check_geom_match(name_a: str, img_a: "nib.Nifti1Image",
                      name_b: str, img_b: "nib.Nifti1Image",
                      config: QCConfig, report: Report) -> None:
    """Shape + affine agreement between two volumes (the alignment guarantee)."""
    sa, sb = _spatial_shape(img_a.shape), _spatial_shape(img_b.shape)
    if sa != sb:
        report.add(f"shape_match_{name_b}", ERROR,
                   f"{name_a} shape {sa} != {name_b} shape {sb}.", {name_a: sa, name_b: sb})
    else:
        report.add(f"shape_match_{name_b}", OK, f"{name_a} and {name_b} share shape {sa}.", sa)

    # Affine agreement. Headline = max abs elementwise diff; we also surface the
    # world-space translation offset in mm, the most intuitive misalignment unit.
    aff_a = np.asarray(img_a.affine, dtype=np.float64)
    aff_b = np.asarray(img_b.affine, dtype=np.float64)
    max_diff = float(np.max(np.abs(aff_a - aff_b)))
    translation_mm = float(np.linalg.norm(aff_a[:3, 3] - aff_b[:3, 3]))
    value = {"max_abs_diff": max_diff, "translation_mm": translation_mm}
    if max_diff > config.affine_atol:
        report.add(f"affine_match_{name_b}", ERROR,
                   f"{name_a} and {name_b} affines differ (max|Δ|={max_diff:.3e}, "
                   f"translation={translation_mm:.3f} mm); volumes are misaligned in world space.",
                   value)
    else:
        report.add(f"affine_match_{name_b}", OK,
                   f"{name_a} and {name_b} affines agree (max|Δ|={max_diff:.3e}).", value)


def _check_labels(name: str, data: np.ndarray, config: QCConfig, report: Report) -> None:
    """Verify a mask/annotation only contains allowed labels."""
    uniques = np.unique(data[np.isfinite(data)])
    allowed = set(config.label_set)
    extra = [float(u) for u in uniques if float(u) not in allowed]
    if extra:
        report.add(f"labels_{name}", ERROR,
                   f"{name} has labels outside allowed {sorted(allowed)}: {extra[:10]}.", extra[:10])
    else:
        report.add(f"labels_{name}", OK,
                   f"{name} labels within allowed {sorted(allowed)}.", [float(u) for u in uniques])


def _check_nonempty(name: str, foreground: np.ndarray, report: Report) -> int:
    """Verify an annotation/mask has at least one active voxel. Returns the count."""
    count = int(np.count_nonzero(foreground))
    if count == 0:
        report.add(f"nonempty_{name}", ERROR, f"{name} is empty (0 active voxels).", 0)
    else:
        report.add(f"nonempty_{name}", OK, f"{name} has {count} active voxels.", count)
    return count


def _check_containment(ann_fg: np.ndarray, mask_fg: np.ndarray,
                       config: QCConfig, report: Report) -> None:
    """Fraction of annotation voxels falling outside the mask (brain)."""
    n_ann = int(np.count_nonzero(ann_fg))
    if n_ann == 0:
        return  # already flagged as empty
    outside = int(np.count_nonzero(ann_fg & ~mask_fg))
    frac = outside / n_ann
    if frac > config.max_annotation_outside_fraction:
        report.add("containment", ERROR,
                   f"{frac:.1%} of annotation voxels fall outside the mask "
                   f"(> {config.max_annotation_outside_fraction:.1%}).", frac)
    else:
        report.add("containment", OK,
                   f"Annotation within mask ({frac:.1%} outside).", frac)


def _check_bbox_border(name: str, foreground: np.ndarray, config: QCConfig, report: Report) -> None:
    """Report the annotation bounding box and warn if it touches the volume border."""
    if not foreground.any():
        return
    coords = np.array(np.where(foreground))
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1)
    bbox = [[int(lo), int(hi)] for lo, hi in zip(mins, maxs)]
    report.meta[f"bbox_{name}"] = bbox
    touches = bool(np.any(mins == 0) or np.any(maxs == np.array(foreground.shape) - 1))
    if touches and config.warn_bbox_touches_border:
        report.add(f"bbox_{name}", WARNING,
                   f"{name} bounding box touches the volume border (possible clipping).", bbox)
    else:
        report.add(f"bbox_{name}", OK, f"{name} bounding box {bbox} inside volume.", bbox)


def _check_components(name: str, foreground: np.ndarray, config: QCConfig, report: Report) -> None:
    """Connected-component count and sizes of an annotation (26-connectivity)."""
    if not foreground.any():
        return
    structure = ndimage.generate_binary_structure(foreground.ndim, foreground.ndim)
    labeled, n = ndimage.label(foreground, structure=structure)
    sizes = np.bincount(labeled.ravel())[1:]  # drop background bin 0
    sizes_sorted = sorted((int(s) for s in sizes), reverse=True)
    report.meta[f"components_{name}"] = {"count": int(n), "sizes": sizes_sorted[:20]}
    tiny = [s for s in sizes_sorted if s < config.min_component_size]
    if n > config.max_components:
        report.add(f"components_{name}", WARNING,
                   f"{name} has {n} connected components (> {config.max_components}); "
                   f"possible fragmentation.", int(n))
    elif tiny:
        report.add(f"components_{name}", WARNING,
                   f"{name} has {len(tiny)} component(s) smaller than "
                   f"{config.min_component_size} voxels (possible spurious blobs).", tiny)
    else:
        report.add(f"components_{name}", OK,
                   f"{name} has {n} connected component(s).", int(n))


def check_pair(image_path: str, mask_path: str, config: Optional[QCConfig] = None) -> Report:
    """
    Check coherence between an image and a mask (brain).

    Verifies matching spatial shape, affine agreement in world space (the
    misalignment that silently breaks training), that the mask only holds allowed
    labels, and that it is non-empty.

    :param image_path: Path to the image NIfTI.
    :param mask_path:  Path to the mask NIfTI.
    :param config:     Thresholds; defaults to :class:`QCConfig`.

    :returns: A :class:`Report` (``kind="pair"``).

    Example
    -------
    >>> from nidataset.qc import check_pair
    >>> rep = check_pair("ct.nii.gz", "brain_mask.nii.gz")
    >>> rep.status
    """
    config = config or QCConfig()
    report = Report(target=f"{image_path} | {mask_path}", kind="pair")

    img = _load(image_path)
    mask = _load(mask_path)
    report.meta["image"] = image_path
    report.meta["mask"] = mask_path

    _check_geom_match("image", img, "mask", mask, config, report)
    mask_data = np.asarray(mask.get_fdata(dtype=np.float64))
    _check_labels("mask", mask_data, config, report)
    _check_nonempty("mask", mask_data > 0, report)

    logger.debug("check_pair(%s, %s) -> %s", image_path, mask_path, report.status)
    return report


def check_triple(image_path: str, mask_path: str, annotation_path: str,
                 config: Optional[QCConfig] = None) -> Report:
    """
    Check coherence between an image, a mask (brain) and an annotation (lesion/bbox).

    Runs the image<->mask checks of :func:`check_pair`, plus image<->annotation
    geometry agreement, annotation labels, non-emptiness, containment of the
    annotation inside the mask, bounding-box bounds and connected-component
    analysis.

    :param image_path:      Path to the image NIfTI.
    :param mask_path:       Path to the mask NIfTI.
    :param annotation_path: Path to the annotation NIfTI.
    :param config:          Thresholds; defaults to :class:`QCConfig`.

    :returns: A :class:`Report` (``kind="triple"``).

    Example
    -------
    >>> from nidataset.qc import check_triple
    >>> rep = check_triple("ct.nii.gz", "brain.nii.gz", "lesion.nii.gz")
    >>> rep.status
    """
    config = config or QCConfig()
    report = Report(target=f"{image_path} | {mask_path} | {annotation_path}", kind="triple")

    img = _load(image_path)
    mask = _load(mask_path)
    ann = _load(annotation_path)
    report.meta["image"] = image_path
    report.meta["mask"] = mask_path
    report.meta["annotation"] = annotation_path

    # image <-> mask, image <-> annotation geometry
    _check_geom_match("image", img, "mask", mask, config, report)
    _check_geom_match("image", img, "annotation", ann, config, report)

    mask_data = np.asarray(mask.get_fdata(dtype=np.float64))
    ann_data = np.asarray(ann.get_fdata(dtype=np.float64))
    mask_fg = mask_data > 0
    ann_fg = ann_data > 0

    _check_labels("mask", mask_data, config, report)
    _check_labels("annotation", ann_data, config, report)
    _check_nonempty("mask", mask_fg, report)
    n_ann = _check_nonempty("annotation", ann_fg, report)

    # Containment / bbox / components only meaningful with shape-aligned, non-empty
    # annotation. Guard on shape so an array-mismatch doesn't raise.
    if n_ann and _spatial_shape(mask.shape) == _spatial_shape(ann.shape):
        _check_containment(ann_fg, mask_fg, config, report)
    _check_bbox_border("annotation", ann_fg, config, report)
    _check_components("annotation", ann_fg, config, report)

    logger.debug("check_triple(...) -> %s", report.status)
    return report
