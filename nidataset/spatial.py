"""
Fast, lossless geometric operations on NIfTI volumes.

Rotate (90-degree multiples), flip, crop to an explicit box, and crop to the
minimal box enclosing the foreground. Every operation updates the affine so the
output stays in the correct world space — a flip or rotation that leaves the
affine untouched silently corrupts left/right and orientation metadata.

These are the "quick shell op" counterparts exposed by the ``nii`` CLI.
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib

from ._helpers import (
    validate_nifti_path,
    ensure_dir,
    strip_nifti_ext,
    run_nifti_dataset,
)

logger = logging.getLogger("nidataset")


def _relabel_affine(affine: np.ndarray, old_shape: Tuple[int, ...], op) -> np.ndarray:
    """
    Return the affine after a pure voxel-relabeling ``op`` (e.g. ``np.flip`` or
    ``np.rot90``) is applied to the data.

    The exact new->old voxel map is *learned* by running ``op`` on a flat-index
    array, so the affine update always matches numpy's own index semantics
    instead of a hand-derived formula that can silently disagree.
    """
    old_shape = tuple(int(s) for s in old_shape)
    lin = np.arange(int(np.prod(old_shape)), dtype=np.int64).reshape(old_shape)
    new = op(lin)  # new[pos] == old flat index that landed at pos
    new_shape = new.shape

    def old_vox(pos: Tuple[int, ...]) -> np.ndarray:
        return np.array(np.unravel_index(int(new[pos]), old_shape), dtype=np.float64)

    origin = old_vox((0, 0, 0))
    M = np.eye(4)
    M[:3, 3] = origin
    for ax in range(3):
        # ponytail: a size-1 new axis has no world extent (only index 0 exists),
        # so its column never places a real voxel - leave it identity.
        if new_shape[ax] < 2:
            continue
        pos = [0, 0, 0]
        pos[ax] = 1
        M[:3, ax] = old_vox(tuple(pos)) - origin
    return affine @ M


def _save_like(data: np.ndarray, affine: np.ndarray, header, out_file: str) -> None:
    nib.save(nib.Nifti1Image(data, affine, header=header), out_file)


# Rotation (90-degree multiples)

def rotate_volume(nii_path: str,
                  output_path: str,
                  k: int = 1,
                  axes: Tuple[int, int] = (0, 1),
                  debug: bool = False) -> str:
    """
    Rotate a NIfTI volume by ``k`` * 90 degrees in the plane spanned by ``axes``.

    Lossless (no interpolation): voxels are relabeled and the affine is updated
    to preserve world space. For arbitrary angles use a resampling-based tool;
    this deliberately supports only 90-degree steps.

    Saves ``<PREFIX>_rot<k>.nii.gz``.

    :param nii_path:    Path to the input NIfTI file.
    :param output_path: Directory for the output.
    :param k:           Number of 90-degree rotations (may be negative).
    :param axes:        The two voxel axes (0/1/2) defining the rotation plane.
    :param debug:       If ``True``, logs the output path.
    :returns: Path to the saved rotated file.

    Example
    -------
    >>> from nidataset.spatial import rotate_volume
    >>> rotate_volume("scan.nii.gz", "out/", k=1, axes=(0, 1))
    """
    validate_nifti_path(nii_path)
    ensure_dir(output_path)
    if len(axes) != 2 or axes[0] == axes[1] or any(a not in (0, 1, 2) for a in axes):
        raise ValueError(f"axes must be two distinct values from (0, 1, 2). Got {axes}.")

    img = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    op = lambda a: np.rot90(a, k=k, axes=axes)  # noqa: E731
    new_data = op(data)
    new_affine = _relabel_affine(img.affine, data.shape, op)

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_rot{k}.nii.gz")
    _save_like(new_data, new_affine, img.header, out_file)
    if debug:
        logger.info("Rotated volume saved at: '%s' (k=%d, axes=%s)", out_file, k, axes)
    return out_file


def rotate_volume_dataset(nii_folder: str,
                          output_path: str,
                          k: int = 1,
                          axes: Tuple[int, int] = (0, 1),
                          debug: bool = False) -> List[str]:
    """Rotate every NIfTI in ``nii_folder`` by ``k`` * 90 degrees. See :func:`rotate_volume`."""
    return run_nifti_dataset(rotate_volume, nii_folder, output_path,
                             desc="Rotating volumes", debug=debug, k=k, axes=axes)


# Flip

def flip_volume(nii_path: str,
                output_path: str,
                axis: int = 0,
                debug: bool = False) -> str:
    """
    Flip (mirror) a NIfTI volume along a voxel ``axis``, updating the affine so
    world space is preserved.

    Saves ``<PREFIX>_flip<axis>.nii.gz``.

    :param nii_path:    Path to the input NIfTI file.
    :param output_path: Directory for the output.
    :param axis:        Voxel axis to flip (0, 1, or 2).
    :param debug:       If ``True``, logs the output path.
    :returns: Path to the saved flipped file.

    Example
    -------
    >>> from nidataset.spatial import flip_volume
    >>> flip_volume("scan.nii.gz", "out/", axis=0)
    """
    validate_nifti_path(nii_path)
    ensure_dir(output_path)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2. Got {axis}.")

    img = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    op = lambda a: np.flip(a, axis=axis)  # noqa: E731
    new_data = op(data)
    new_affine = _relabel_affine(img.affine, data.shape, op)

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_flip{axis}.nii.gz")
    _save_like(new_data, new_affine, img.header, out_file)
    if debug:
        logger.info("Flipped volume saved at: '%s' (axis=%d)", out_file, axis)
    return out_file


def flip_volume_dataset(nii_folder: str,
                        output_path: str,
                        axis: int = 0,
                        debug: bool = False) -> List[str]:
    """Flip every NIfTI in ``nii_folder`` along ``axis``. See :func:`flip_volume`."""
    return run_nifti_dataset(flip_volume, nii_folder, output_path,
                             desc="Flipping volumes", debug=debug, axis=axis)


# Crop to an explicit box

def crop_volume(nii_path: str,
                output_path: str,
                bbox: Tuple[int, int, int, int, int, int],
                debug: bool = False) -> str:
    """
    Crop a NIfTI volume to an explicit voxel box, shifting the affine origin so
    the retained voxels keep their world coordinates.

    Saves ``<PREFIX>_cropped.nii.gz``.

    :param nii_path:    Path to the input NIfTI file.
    :param output_path: Directory for the output.
    :param bbox:        ``(x0, x1, y0, y1, z0, z1)`` half-open voxel bounds
                        (``x0:x1`` etc., like numpy slicing).
    :param debug:       If ``True``, logs the output path.
    :returns: Path to the saved cropped file.

    Example
    -------
    >>> from nidataset.spatial import crop_volume
    >>> crop_volume("scan.nii.gz", "out/", bbox=(10, 100, 10, 100, 5, 60))
    """
    validate_nifti_path(nii_path)
    ensure_dir(output_path)

    img = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    x0, x1, y0, y1, z0, z1 = bbox
    bounds = [(x0, x1), (y0, y1), (z0, z1)]
    for ax, (lo, hi) in enumerate(bounds):
        if not (0 <= lo < hi <= data.shape[ax]):
            raise ValueError(
                f"bbox out of range on axis {ax}: got [{lo}, {hi}) for size {data.shape[ax]}."
            )

    cropped = data[x0:x1, y0:y1, z0:z1]
    new_affine = img.affine.copy()
    new_affine[:3, 3] = img.affine[:3, :3] @ np.array([x0, y0, z0]) + img.affine[:3, 3]

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_cropped.nii.gz")
    _save_like(cropped, new_affine, img.header, out_file)
    if debug:
        logger.info("Cropped volume saved at: '%s' (bbox=%s)", out_file, bbox)
    return out_file


# Crop to the minimal box enclosing the foreground

def crop_to_content(nii_path: str,
                    output_path: str,
                    threshold: Optional[float] = None,
                    margin: int = 0,
                    debug: bool = False) -> str:
    """
    Crop a NIfTI volume to the minimal box enclosing its foreground (the minimum
    enclosing rectangle), trimming empty borders. The affine is shifted so the
    kept voxels keep their world coordinates.

    Foreground is ``data != 0`` when ``threshold`` is ``None``, else ``data > threshold``.

    Saves ``<PREFIX>_content.nii.gz``.

    :param nii_path:    Path to the input NIfTI file.
    :param output_path: Directory for the output.
    :param threshold:   Foreground cutoff; ``None`` keeps all non-zero voxels.
    :param margin:      Voxels of padding kept around the box (clamped to bounds).
    :param debug:       If ``True``, logs the output path and box.
    :returns: Path to the saved cropped file.

    :raises ValueError: If no voxel passes the foreground test (empty volume).

    Example
    -------
    >>> from nidataset.spatial import crop_to_content
    >>> crop_to_content("scan.nii.gz", "out/", threshold=0, margin=2)
    """
    validate_nifti_path(nii_path)
    ensure_dir(output_path)

    img = nib.load(nii_path)
    data = np.asarray(img.dataobj)
    fg = data != 0 if threshold is None else data > threshold
    if not fg.any():
        raise ValueError("No foreground voxels found; cannot crop to content.")

    starts, stops = [], []
    for ax in range(3):
        axis_any = fg.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.where(axis_any)[0]
        lo = max(int(idx[0]) - margin, 0)
        hi = min(int(idx[-1]) + 1 + margin, data.shape[ax])
        starts.append(lo)
        stops.append(hi)

    bbox = (starts[0], stops[0], starts[1], stops[1], starts[2], stops[2])
    cropped = data[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]]
    new_affine = img.affine.copy()
    new_affine[:3, 3] = img.affine[:3, :3] @ np.array(starts) + img.affine[:3, 3]

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_content.nii.gz")
    _save_like(cropped, new_affine, img.header, out_file)
    if debug:
        logger.info("Content-cropped volume saved at: '%s' (bbox=%s)", out_file, bbox)
    return out_file


def crop_to_content_dataset(nii_folder: str,
                            output_path: str,
                            threshold: Optional[float] = None,
                            margin: int = 0,
                            debug: bool = False) -> List[str]:
    """Crop every NIfTI in ``nii_folder`` to its foreground box. See :func:`crop_to_content`."""
    return run_nifti_dataset(crop_to_content, nii_folder, output_path,
                             desc="Cropping to content", debug=debug,
                             threshold=threshold, margin=margin)
