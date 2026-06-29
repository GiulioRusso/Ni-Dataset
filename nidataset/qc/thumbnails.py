"""
Visual QC thumbnails: a PNG grid of central orthogonal slices.

One row per item, three columns (sagittal / coronal / axial central slices), so
hundreds of volumes can be eyeballed at once. For pairs/triples the mask and
annotation are overlaid in translucent colour on the image, making spatial
misalignment obvious at a glance.

Intensity is normalised on the 1st/99th percentiles (not raw min/max) so a few
extreme voxels don't render every thumbnail black. Uses only Pillow + numpy +
nibabel and the package's existing colormap helper — no new dependencies.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import nibabel as nib
from PIL import Image

from .._helpers import ensure_dir, strip_nifti_ext, logger

# Overlay colours (RGB) and blend strength.
_MASK_RGB = (60, 200, 90)
_ANN_RGB = (235, 60, 60)
_OVERLAY_ALPHA = 0.45

_PLANES = ("sagittal", "coronal", "axial")  # axes 0, 1, 2


def _robust_norm(sl: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Normalize a 2D slice to uint8 using percentile clipping (robust to outliers)."""
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        return np.zeros(sl.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(sl.shape, dtype=np.uint8)
    out = np.clip((sl - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def _central_slice(data: np.ndarray, axis: int) -> np.ndarray:
    """Return the central slice along *axis*, oriented for upright display."""
    idx = data.shape[axis] // 2
    sl = np.take(data, idx, axis=axis)
    # Transpose+flip so rows go top->bottom as expected in radiological-ish view.
    return np.flipud(sl.T)


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    """Stack a uint8 grayscale slice into an RGB array."""
    return np.stack([gray, gray, gray], axis=-1)


def _overlay(rgb: np.ndarray, fg: np.ndarray, color: Sequence[int]) -> np.ndarray:
    """Alpha-blend *color* onto *rgb* where boolean mask *fg* is true."""
    if not fg.any():
        return rgb
    out = rgb.astype(np.float32)
    col = np.array(color, dtype=np.float32)
    out[fg] = (1 - _OVERLAY_ALPHA) * out[fg] + _OVERLAY_ALPHA * col
    return out.astype(np.uint8)


def _cell(image: np.ndarray, axis: int, cell_size: int,
          mask: Optional[np.ndarray], ann: Optional[np.ndarray]) -> Image.Image:
    """Render one (image[, overlays]) plane as a square Pillow image."""
    gray = _robust_norm(_central_slice(image, axis))
    rgb = _to_rgb(gray)
    if mask is not None and mask.shape == image.shape:
        rgb = _overlay(rgb, _central_slice(mask, axis) > 0, _MASK_RGB)
    if ann is not None and ann.shape == image.shape:
        rgb = _overlay(rgb, _central_slice(ann, axis) > 0, _ANN_RGB)
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((cell_size, cell_size), Image.NEAREST)


def _load_spatial(path: Optional[str]) -> Optional[np.ndarray]:
    """Load a NIfTI as a 3D float array (first volume if 4D); ``None`` passthrough."""
    if path is None:
        return None
    data = np.asarray(nib.load(path).get_fdata(dtype=np.float64))
    while data.ndim > 3:
        data = data[..., 0]
    return data


def thumbnail_grid(specs: List[Dict[str, Optional[str]]], output_path: str,
                   cell_size: int = 160) -> str:
    """
    Render a PNG grid of central orthogonal slices, one row per item.

    Each spec is a dict with keys ``image`` (required) and optional ``mask`` /
    ``annotation`` (overlaid translucently) and ``label`` (row caption fallback is
    the image filename). Columns are sagittal, coronal and axial central slices.

    :param specs:       List of ``{"image", "mask"?, "annotation"?, "label"?}``.
    :param output_path: Output ``.png`` path (parent dirs are created).
    :param cell_size:   Pixel size of each square plane cell.

    :returns: The written PNG path.

    Example
    -------
    >>> from nidataset.qc import thumbnail_grid
    >>> thumbnail_grid([{"image": "ct.nii.gz", "mask": "brain.nii.gz"}], "qc.png")
    """
    if not specs:
        raise ValueError("thumbnail_grid: no specs provided.")
    ensure_dir(os.path.dirname(output_path) or ".")

    n_cols = len(_PLANES)
    rows = []
    for spec in specs:
        image_path = spec.get("image")
        if not image_path:
            continue
        try:
            image = _load_spatial(image_path)
            mask = _load_spatial(spec.get("mask"))
            ann = _load_spatial(spec.get("annotation"))
        except Exception as exc:  # keep the grid going even if one file is bad
            logger.warning("thumbnail_grid: skipping %s (%s)", image_path, exc)
            continue
        cells = [_cell(image, axis, cell_size, mask, ann) for axis in range(n_cols)]
        row = Image.new("RGB", (cell_size * n_cols, cell_size), (0, 0, 0))
        for c, cell in enumerate(cells):
            row.paste(cell, (c * cell_size, 0))
        rows.append(row)

    if not rows:
        raise ValueError("thumbnail_grid: no readable images among specs.")

    grid = Image.new("RGB", (cell_size * n_cols, cell_size * len(rows)), (0, 0, 0))
    for r, row in enumerate(rows):
        grid.paste(row, (0, r * cell_size))
    grid.save(output_path)
    logger.info("thumbnail_grid saved: '%s' (%d rows)", output_path, len(rows))
    return output_path


def _label(spec: Dict[str, Optional[str]]) -> str:
    """Best-effort row label for a spec (used by the CLI)."""
    if spec.get("label"):
        return str(spec["label"])
    img = spec.get("image") or ""
    return strip_nifti_ext(os.path.basename(img))
