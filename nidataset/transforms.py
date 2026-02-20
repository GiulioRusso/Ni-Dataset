"""
Intensity transforms and spatial utility routines for NIfTI volumes.

Provides windowing, intensity normalization, resampling to reference,
and format conversion helpers.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

from ._helpers import (
    validate_nifti_path,
    list_nifti_files,
    ensure_dir,
    strip_nifti_ext,
)

logger = logging.getLogger("nidataset")



# Intensity normalization

def intensity_normalization(nii_path: str,
                            output_path: str,
                            method: str = "zscore",
                            percentile_range: Tuple[float, float] = (1.0, 99.0),
                            reference_path: Optional[str] = None,
                            debug: bool = False) -> str:
    """
    Normalize the intensity values of a 3D NIfTI volume.

    Supported methods:
        - ``"zscore"``       – Zero-mean, unit-variance normalization.
        - ``"minmax"``       – Scale to [0, 1] range.
        - ``"percentile"``   – Clip to the given percentile range, then min-max.
        - ``"histogram"``    – Match histogram to a reference volume.

    Saves:
        ``<PREFIX>_normalized.nii.gz``

    :param nii_path:         Path to the input NIfTI file.
    :param output_path:      Directory where the output will be saved.
    :param method:           Normalization method (default: ``"zscore"``).
    :param percentile_range: Tuple ``(low, high)`` percentiles for ``"percentile"`` method.
    :param reference_path:   Reference NIfTI for ``"histogram"`` matching.
    :param debug:            If ``True``, logs details.

    :returns: Path to the saved normalized file.

    :raises ValueError: If method is unknown or reference is missing for histogram.

    Example
    -------
    >>> from nidataset.transforms import intensity_normalization
    >>> intensity_normalization("scan.nii.gz", "output/", method="zscore")
    """

    valid_methods = {"zscore", "minmax", "percentile", "histogram"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {sorted(valid_methods)}.")

    validate_nifti_path(nii_path)
    ensure_dir(output_path)

    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata().astype(np.float64)

    if method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            data = (data - mean) / std
        else:
            logger.warning("Std is zero; skipping z-score normalization.")

    elif method == "minmax":
        dmin, dmax = np.min(data), np.max(data)
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)

    elif method == "percentile":
        low_val = np.percentile(data, percentile_range[0])
        high_val = np.percentile(data, percentile_range[1])
        data = np.clip(data, low_val, high_val)
        if high_val > low_val:
            data = (data - low_val) / (high_val - low_val)

    elif method == "histogram":
        if reference_path is None:
            raise ValueError("reference_path is required for histogram matching.")
        validate_nifti_path(reference_path)
        ref_data = nib.load(reference_path).get_fdata().ravel()

        # Histogram matching via sorted-quantile mapping
        src_sorted = np.sort(data.ravel())
        ref_sorted = np.sort(ref_data)
        interp_values = np.interp(
            np.linspace(0, 1, len(src_sorted)),
            np.linspace(0, 1, len(ref_sorted)),
            ref_sorted,
        )
        mapping = dict(zip(src_sorted, interp_values))
        sort_idx = np.argsort(data.ravel())
        matched = np.empty_like(data.ravel())
        matched[sort_idx] = interp_values
        data = matched.reshape(data.shape)

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_normalized.nii.gz")
    nib.save(nib.Nifti1Image(data.astype(np.float32), nii_img.affine, nii_img.header), out_file)

    if debug:
        logger.info("Normalized volume saved at: '%s' (method=%s)", out_file, method)
    return out_file


def intensity_normalization_dataset(nii_folder: str,
                                    output_path: str,
                                    method: str = "zscore",
                                    percentile_range: Tuple[float, float] = (1.0, 99.0),
                                    reference_path: Optional[str] = None,
                                    debug: bool = False) -> List[str]:
    """
    Apply intensity normalization to all NIfTI files in a folder.

    :param nii_folder:       Folder containing NIfTI files.
    :param output_path:      Output directory.
    :param method:           Normalization method (see ``intensity_normalization``).
    :param percentile_range: Percentile range for ``"percentile"`` method.
    :param reference_path:   Reference NIfTI for ``"histogram"`` matching.
    :param debug:            If ``True``, logs details.

    :returns: List of output file paths.

    Example
    -------
    >>> intensity_normalization_dataset("scans/", "output/", method="minmax")
    """

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)
    results = []

    for nii_file in tqdm(nii_files, desc="Normalizing intensities", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)
        try:
            out = intensity_normalization(
                nii_path, output_path, method=method,
                percentile_range=percentile_range,
                reference_path=reference_path, debug=debug,
            )
            results.append(out)
        except Exception as e:
            logger.warning("Error normalizing %s: %s", nii_file, e)

    return results


# CT Windowing

# Common CT window presets: (window_center, window_width)
CT_WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "brain": (40, 80),
    "subdural": (75, 215),
    "stroke": (40, 40),
    "bone": (480, 2500),
    "soft_tissue": (50, 350),
    "lung": (-600, 1500),
    "liver": (60, 160),
    "mediastinum": (50, 350),
}


def windowing(nii_path: str,
              output_path: str,
              window_center: Optional[float] = None,
              window_width: Optional[float] = None,
              preset: Optional[str] = None,
              normalize: bool = True,
              debug: bool = False) -> str:
    """
    Apply CT windowing (window center + window width) to a NIfTI volume.

    Either specify ``window_center`` and ``window_width`` directly, or use a
    named ``preset`` (e.g., ``"brain"``, ``"bone"``, ``"lung"``).

    Available presets: brain, subdural, stroke, bone, soft_tissue, lung,
    liver, mediastinum.

    Saves:
        ``<PREFIX>_windowed.nii.gz``

    :param nii_path:       Path to the input NIfTI file.
    :param output_path:    Directory for the output.
    :param window_center:  Center of the window (Hounsfield units).
    :param window_width:   Width of the window.
    :param preset:         Named window preset (overrides center/width).
    :param normalize:      If ``True``, scale windowed values to [0, 1].
    :param debug:          If ``True``, logs details.

    :returns: Path to the saved windowed file.

    Example
    -------
    >>> windowing("ct_scan.nii.gz", "output/", preset="brain")
    >>> windowing("ct_scan.nii.gz", "output/", window_center=40, window_width=80)
    """

    if preset is not None:
        if preset not in CT_WINDOW_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(CT_WINDOW_PRESETS.keys())}")
        window_center, window_width = CT_WINDOW_PRESETS[preset]
    elif window_center is None or window_width is None:
        raise ValueError("Specify either (window_center, window_width) or a preset name.")

    validate_nifti_path(nii_path)
    ensure_dir(output_path)

    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata().astype(np.float64)

    low = window_center - window_width / 2.0
    high = window_center + window_width / 2.0
    data = np.clip(data, low, high)

    if normalize:
        data = (data - low) / (high - low)

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    suffix = f"_windowed_{preset}" if preset else "_windowed"
    out_file = os.path.join(output_path, f"{prefix}{suffix}.nii.gz")
    nib.save(nib.Nifti1Image(data.astype(np.float32), nii_img.affine, nii_img.header), out_file)

    if debug:
        logger.info("Windowed volume saved at: '%s' (center=%.1f, width=%.1f)",
                     out_file, window_center, window_width)
    return out_file


def windowing_dataset(nii_folder: str,
                      output_path: str,
                      window_center: Optional[float] = None,
                      window_width: Optional[float] = None,
                      preset: Optional[str] = None,
                      normalize: bool = True,
                      debug: bool = False) -> List[str]:
    """
    Apply CT windowing to all NIfTI files in a folder.

    :param nii_folder:    Folder containing NIfTI files.
    :param output_path:   Output directory.
    :param window_center: Center of the window.
    :param window_width:  Width of the window.
    :param preset:        Named window preset (see ``windowing``).
    :param normalize:     If ``True``, scale to [0, 1].
    :param debug:         If ``True``, logs details.

    :returns: List of output file paths.

    Example
    -------
    >>> windowing_dataset("scans/", "output/", preset="brain")
    """

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)
    results = []

    for nii_file in tqdm(nii_files, desc="Applying windowing", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)
        try:
            out = windowing(
                nii_path, output_path,
                window_center=window_center, window_width=window_width,
                preset=preset, normalize=normalize, debug=debug,
            )
            results.append(out)
        except Exception as e:
            logger.warning("Error windowing %s: %s", nii_file, e)

    return results


# Resample to reference

def resample_to_reference(nii_path: str,
                          reference_path: str,
                          output_path: str,
                          interpolation: str = "linear",
                          debug: bool = False) -> str:
    """
    Resample a NIfTI volume to match the spatial grid of a reference volume.

    The output volume will have the same origin, spacing, direction, and
    size as the reference. This is useful when combining volumes from
    different sources that need to be in the same physical space.

    Saves:
        ``<PREFIX>_resampled_to_ref.nii.gz``

    :param nii_path:       Path to the input NIfTI file.
    :param reference_path: Path to the reference NIfTI file.
    :param output_path:    Directory for the output.
    :param interpolation:  ``"linear"`` (default), ``"nearest"``, or ``"bspline"``.
    :param debug:          If ``True``, logs details.

    :returns: Path to the saved resampled file.

    Example
    -------
    >>> resample_to_reference("moving.nii.gz", "fixed.nii.gz", "output/")
    """

    interp_map = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }
    if interpolation not in interp_map:
        raise ValueError(f"Unknown interpolation '{interpolation}'. Choose from {sorted(interp_map)}.")

    validate_nifti_path(nii_path)
    validate_nifti_path(reference_path)
    ensure_dir(output_path)

    moving = sitk.ReadImage(nii_path)
    reference = sitk.ReadImage(reference_path)

    resampled = sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        interp_map[interpolation],
        0.0,
        moving.GetPixelID(),
    )

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_resampled_to_ref.nii.gz")
    sitk.WriteImage(resampled, out_file)

    if debug:
        logger.info("Resampled to reference saved at: '%s'", out_file)
    return out_file


def resample_to_reference_dataset(nii_folder: str,
                                  reference_path: str,
                                  output_path: str,
                                  interpolation: str = "linear",
                                  debug: bool = False) -> List[str]:
    """
    Resample all NIfTI files in a folder to match a reference volume.

    :param nii_folder:     Folder containing NIfTI files.
    :param reference_path: Path to the reference NIfTI file.
    :param output_path:    Output directory.
    :param interpolation:  ``"linear"``, ``"nearest"``, or ``"bspline"``.
    :param debug:          If ``True``, logs details.

    :returns: List of output file paths.

    Example
    -------
    >>> resample_to_reference_dataset("scans/", "template.nii.gz", "output/")
    """

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)
    results = []

    for nii_file in tqdm(nii_files, desc="Resampling to reference", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)
        try:
            out = resample_to_reference(nii_path, reference_path, output_path,
                                        interpolation=interpolation, debug=debug)
            results.append(out)
        except Exception as e:
            logger.warning("Error resampling %s: %s", nii_file, e)

    return results


# Apply generic transform

def apply_transform(nii_path: str,
                    transform_path: str,
                    reference_path: str,
                    output_path: str,
                    interpolation: str = "linear",
                    debug: bool = False) -> str:
    """
    Apply a saved spatial transformation to any NIfTI volume.

    This is a generic version of ``register_mask`` / ``register_annotation``
    without filename suffix restrictions.

    Saves:
        ``<PREFIX>_transformed.nii.gz``

    :param nii_path:        Path to the input NIfTI file.
    :param transform_path:  Path to the ``.tfm`` transformation file.
    :param reference_path:  Path to the reference NIfTI defining target space.
    :param output_path:     Directory for the output.
    :param interpolation:   ``"linear"`` (default), ``"nearest"``, or ``"bspline"``.
    :param debug:           If ``True``, logs details.

    :returns: Path to the saved transformed file.

    Example
    -------
    >>> apply_transform("any_volume.nii.gz", "transform.tfm", "reference.nii.gz", "output/")
    """

    interp_map = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }
    if interpolation not in interp_map:
        raise ValueError(f"Unknown interpolation '{interpolation}'. Choose from {sorted(interp_map)}.")

    validate_nifti_path(nii_path)
    validate_nifti_path(reference_path)
    if not os.path.isfile(transform_path):
        raise FileNotFoundError(f"Transform file not found: '{transform_path}'")

    ensure_dir(output_path)

    moving = sitk.ReadImage(nii_path, sitk.sitkFloat32)
    reference = sitk.ReadImage(reference_path, sitk.sitkFloat32)
    transform = sitk.ReadTransform(transform_path)

    transformed = sitk.Resample(
        moving, reference, transform,
        interp_map[interpolation], 0.0, moving.GetPixelID(),
    )

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_transformed.nii.gz")
    sitk.WriteImage(transformed, out_file)

    if debug:
        logger.info("Transformed volume saved at: '%s'", out_file)
    return out_file


# Format conversion

def nifti_to_numpy(nii_path: str,
                   output_path: str,
                   compressed: bool = True,
                   debug: bool = False) -> str:
    """
    Convert a NIfTI volume to a NumPy ``.npy`` or ``.npz`` file.

    :param nii_path:    Path to the input NIfTI file.
    :param output_path: Directory for the output.
    :param compressed:  If ``True``, saves as ``.npz``. Otherwise ``.npy``.
    :param debug:       If ``True``, logs details.

    :returns: Path to the saved NumPy file.

    Example
    -------
    >>> nifti_to_numpy("scan.nii.gz", "output/")
    """

    validate_nifti_path(nii_path)
    ensure_dir(output_path)

    data = nib.load(nii_path).get_fdata()
    prefix = strip_nifti_ext(os.path.basename(nii_path))

    if compressed:
        out_file = os.path.join(output_path, f"{prefix}.npz")
        np.savez_compressed(out_file, data=data)
    else:
        out_file = os.path.join(output_path, f"{prefix}.npy")
        np.save(out_file, data)

    if debug:
        logger.info("NumPy file saved at: '%s'", out_file)
    return out_file


def numpy_to_nifti(npy_path: str,
                   output_path: str,
                   affine: Optional[np.ndarray] = None,
                   reference_nifti: Optional[str] = None,
                   debug: bool = False) -> str:
    """
    Convert a NumPy ``.npy`` or ``.npz`` file to NIfTI format.

    :param npy_path:         Path to the .npy or .npz file.
    :param output_path:      Directory for the output.
    :param affine:           4x4 affine matrix. If None, uses identity.
    :param reference_nifti:  Optional NIfTI path to copy affine/header from.
    :param debug:            If ``True``, logs details.

    :returns: Path to the saved NIfTI file.

    Example
    -------
    >>> numpy_to_nifti("data.npz", "output/", reference_nifti="original.nii.gz")
    """

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"File not found: '{npy_path}'")

    ensure_dir(output_path)

    if npy_path.endswith(".npz"):
        npz = np.load(npy_path)
        data = npz[list(npz.keys())[0]]
    else:
        data = np.load(npy_path)

    ref_header = None
    if reference_nifti is not None:
        validate_nifti_path(reference_nifti)
        ref_img = nib.load(reference_nifti)
        affine = ref_img.affine
        ref_header = ref_img.header
    elif affine is None:
        affine = np.eye(4)

    prefix = os.path.splitext(os.path.splitext(os.path.basename(npy_path))[0])[0]
    out_file = os.path.join(output_path, f"{prefix}.nii.gz")
    nib.save(nib.Nifti1Image(data, affine, header=ref_header), out_file)

    if debug:
        logger.info("NIfTI file saved at: '%s'", out_file)
    return out_file
