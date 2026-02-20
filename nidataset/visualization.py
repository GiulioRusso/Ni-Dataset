"""
Visualization routines for NIfTI volumes.

Provides mask overlay generation and slice montage creation.
"""

import os
import logging
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm
from PIL import Image

from ._helpers import (
    validate_nifti_path,
    list_nifti_files,
    ensure_dir,
    strip_nifti_ext,
    validate_view,
)

logger = logging.getLogger("nidataset")


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 255] uint8 range."""
    dmin, dmax = np.min(data), np.max(data)
    if dmax > dmin:
        return ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    return np.zeros_like(data, dtype=np.uint8)


def _get_colormap(name: str, n: int = 256) -> np.ndarray:
    """Return an Nx3 uint8 colormap array without requiring matplotlib at import time."""
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(name, n)
        colors = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
        return colors
    except ImportError:
        # Fallback: simple red colormap
        colors = np.zeros((n, 3), dtype=np.uint8)
        colors[:, 0] = np.linspace(0, 255, n).astype(np.uint8)
        return colors


def overlay_mask_on_volume(nii_path: str,
                           mask_path: str,
                           output_path: str,
                           view: str = "axial",
                           alpha: float = 0.4,
                           colormap: str = "jet",
                           output_format: str = "png",
                           debug: bool = False) -> List[str]:
    """
    Create colored overlay images of a segmentation mask on a grayscale volume.

    For each slice along the chosen anatomical axis, saves an RGB image with
    the mask blended on top of the grayscale volume using the specified
    colormap and alpha.

    Saves:
        ``<PREFIX>_overlay_<VIEW>_<NNN>.<FORMAT>``

    :param nii_path:       Path to the grayscale NIfTI volume.
    :param mask_path:      Path to the segmentation mask NIfTI volume.
    :param output_path:    Directory for output images.
    :param view:           ``"axial"``, ``"coronal"``, or ``"sagittal"``.
    :param alpha:          Mask overlay opacity (0.0 = transparent, 1.0 = opaque).
    :param colormap:       Matplotlib colormap name (default: ``"jet"``).
    :param output_format:  Image format: ``"png"``, ``"tif"``, or ``"jpg"``.
    :param debug:          If ``True``, logs details.

    :returns: List of saved overlay image paths.

    Example
    -------
    >>> overlay_mask_on_volume("brain.nii.gz", "mask.nii.gz", "output/",
    ...                        view="axial", alpha=0.4, colormap="hot")
    """

    validate_nifti_path(nii_path)
    validate_nifti_path(mask_path)
    validate_view(view)
    ensure_dir(output_path)

    vol_data = nib.load(nii_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()

    if vol_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: volume {vol_data.shape} vs mask {mask_data.shape}."
        )

    view_mapping = {"axial": 2, "coronal": 1, "sagittal": 0}
    axis = view_mapping[view]

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    cmap_colors = _get_colormap(colormap)

    saved_paths: List[str] = []
    num_slices = vol_data.shape[axis]

    for i in tqdm(range(num_slices), desc=f"Overlay {prefix} ({view})", unit="slice"):
        if axis == 2:
            vol_slice = vol_data[:, :, i]
            mask_slice = mask_data[:, :, i]
        elif axis == 1:
            vol_slice = vol_data[:, i, :]
            mask_slice = mask_data[:, i, :]
        else:
            vol_slice = vol_data[i, :, :]
            mask_slice = mask_data[i, :, :]

        # Normalize grayscale to uint8
        gray = _normalize_to_uint8(vol_slice)
        rgb = np.stack([gray, gray, gray], axis=-1)

        # Where mask is nonzero, blend color
        mask_binary = mask_slice > 0
        if np.any(mask_binary):
            mask_norm = _normalize_to_uint8(mask_slice)
            color_overlay = cmap_colors[mask_norm]

            rgb[mask_binary] = (
                (1 - alpha) * rgb[mask_binary] + alpha * color_overlay[mask_binary]
            ).astype(np.uint8)

        fname = f"{prefix}_overlay_{view}_{str(i).zfill(3)}.{output_format}"
        fpath = os.path.join(output_path, fname)
        Image.fromarray(rgb).save(fpath)
        saved_paths.append(fpath)

    if debug:
        logger.info("Overlay images saved to: '%s' (%d slices)", output_path, num_slices)

    return saved_paths


def overlay_mask_on_volume_dataset(nii_folder: str,
                                   mask_folder: str,
                                   output_path: str,
                                   view: str = "axial",
                                   alpha: float = 0.4,
                                   colormap: str = "jet",
                                   output_format: str = "png",
                                   debug: bool = False) -> int:
    """
    Create mask overlay images for all matching NIfTI pairs in two folders.

    Files are matched by filename. Each case's overlays are saved in a
    subdirectory named after the case.

    :param nii_folder:    Folder with grayscale NIfTI volumes.
    :param mask_folder:   Folder with mask NIfTI volumes.
    :param output_path:   Output directory.
    :param view:          Anatomical view.
    :param alpha:         Overlay opacity.
    :param colormap:      Matplotlib colormap name.
    :param output_format: Image format.
    :param debug:         If ``True``, logs details.

    :returns: Total number of overlay images generated.

    Example
    -------
    >>> overlay_mask_on_volume_dataset("scans/", "masks/", "output/overlays/")
    """

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)
    total = 0

    for nii_file in tqdm(nii_files, desc="Generating overlays", unit="case"):
        nii_path = os.path.join(nii_folder, nii_file)
        mask_path = os.path.join(mask_folder, nii_file)

        if not os.path.isfile(mask_path):
            logger.warning("No matching mask for '%s', skipping.", nii_file)
            continue

        prefix = strip_nifti_ext(nii_file)
        case_dir = os.path.join(output_path, prefix)

        try:
            paths = overlay_mask_on_volume(
                nii_path, mask_path, case_dir,
                view=view, alpha=alpha, colormap=colormap,
                output_format=output_format, debug=debug,
            )
            total += len(paths)
        except Exception as e:
            logger.warning("Error generating overlay for %s: %s", nii_file, e)

    if debug:
        logger.info("Total overlay images generated: %d", total)

    return total


def create_slice_montage(nii_path: str,
                         output_path: str,
                         view: str = "axial",
                         num_slices: int = 16,
                         cols: int = 4,
                         debug: bool = False) -> str:
    """
    Create a montage (grid) of evenly-spaced slices from a NIfTI volume.

    Saves:
        ``<PREFIX>_montage_<VIEW>.png``

    :param nii_path:    Path to the NIfTI file.
    :param output_path: Directory for the output image.
    :param view:        ``"axial"``, ``"coronal"``, or ``"sagittal"``.
    :param num_slices:  Number of slices to include in the montage.
    :param cols:        Number of columns in the grid.
    :param debug:       If ``True``, logs details.

    :returns: Path to the saved montage image.

    Example
    -------
    >>> create_slice_montage("brain.nii.gz", "output/", view="axial", num_slices=20)
    """

    validate_nifti_path(nii_path)
    validate_view(view)
    ensure_dir(output_path)

    data = nib.load(nii_path).get_fdata()
    view_mapping = {"axial": 2, "coronal": 1, "sagittal": 0}
    axis = view_mapping[view]
    total = data.shape[axis]

    indices = np.linspace(0, total - 1, num_slices, dtype=int)
    rows = (num_slices + cols - 1) // cols

    slices = []
    for idx in indices:
        if axis == 2:
            s = data[:, :, idx]
        elif axis == 1:
            s = data[:, idx, :]
        else:
            s = data[idx, :, :]
        slices.append(_normalize_to_uint8(s))

    h, w = slices[0].shape
    montage = np.zeros((rows * h, cols * w), dtype=np.uint8)

    for i, s in enumerate(slices):
        r, c = divmod(i, cols)
        montage[r * h:(r + 1) * h, c * w:(c + 1) * w] = s

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_montage_{view}.png")
    Image.fromarray(montage).save(out_file)

    if debug:
        logger.info("Montage saved at: '%s' (%d slices, %dx%d grid)", out_file, num_slices, rows, cols)

    return out_file
