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
    run_nifti_dataset,
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


# Linear intensity rescale

def rescale_intensity(nii_path: str,
                      output_path: str,
                      out_min: float = 0.0,
                      out_max: float = 1.0,
                      in_min: Optional[float] = None,
                      in_max: Optional[float] = None,
                      debug: bool = False) -> str:
    """
    Linearly rescale a NIfTI volume's intensities to ``[out_min, out_max]``.

    Values are clipped to the input range (``[in_min, in_max]``, defaulting to the
    volume's own min/max) before mapping. This is the plain linear counterpart to
    :func:`intensity_normalization` (which offers z-score / percentile / histogram
    modes) — use it when you just want a fixed output range.

    Saves ``<PREFIX>_rescaled.nii.gz``.

    :param nii_path:    Path to the input NIfTI file.
    :param output_path: Directory for the output.
    :param out_min:     Lower bound of the output range.
    :param out_max:     Upper bound of the output range.
    :param in_min:      Lower bound of the input range (default: data min).
    :param in_max:      Upper bound of the input range (default: data max).
    :param debug:       If ``True``, logs the output path.
    :returns: Path to the saved rescaled file.

    Example
    -------
    >>> from nidataset.transforms import rescale_intensity
    >>> rescale_intensity("scan.nii.gz", "out/", out_min=0, out_max=255)
    """
    if out_max <= out_min:
        raise ValueError(f"out_max must be > out_min. Got ({out_min}, {out_max}).")

    validate_nifti_path(nii_path)
    ensure_dir(output_path)

    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata().astype(np.float64)

    lo = np.min(data) if in_min is None else in_min
    hi = np.max(data) if in_max is None else in_max
    if hi <= lo:
        raise ValueError(f"Input range is empty (in_min={lo}, in_max={hi}).")

    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo)
    data = data * (out_max - out_min) + out_min

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    out_file = os.path.join(output_path, f"{prefix}_rescaled.nii.gz")
    nib.save(nib.Nifti1Image(data.astype(np.float32), nii_img.affine, nii_img.header), out_file)

    if debug:
        logger.info("Rescaled volume saved at: '%s' ([%.3g, %.3g] -> [%.3g, %.3g])",
                    out_file, lo, hi, out_min, out_max)
    return out_file


def rescale_intensity_dataset(nii_folder: str,
                              output_path: str,
                              out_min: float = 0.0,
                              out_max: float = 1.0,
                              in_min: Optional[float] = None,
                              in_max: Optional[float] = None,
                              debug: bool = False) -> List[str]:
    """Rescale every NIfTI in ``nii_folder``. See :func:`rescale_intensity`."""
    return run_nifti_dataset(rescale_intensity, nii_folder, output_path,
                             desc="Rescaling intensities", debug=debug,
                             out_min=out_min, out_max=out_max,
                             in_min=in_min, in_max=in_max)


# DICOM <-> NIfTI conversion

def dicom_to_nifti(dicom_dir: str,
                   output_path: str,
                   debug: bool = False) -> str:
    """
    Convert a directory of DICOM slices (one series) into a single NIfTI volume.

    Reads the series with SimpleITK's GDCM reader, preserving spacing, origin,
    and orientation. The output is named after the DICOM directory.

    Saves ``<DIRNAME>.nii.gz``.

    :param dicom_dir:   Directory holding the ``.dcm`` slices of one series.
    :param output_path: Directory for the output NIfTI.
    :param debug:       If ``True``, logs the output path.
    :returns: Path to the saved NIfTI file.

    :raises FileNotFoundError: If ``dicom_dir`` is not a directory or has no series.

    Example
    -------
    >>> from nidataset.transforms import dicom_to_nifti
    >>> dicom_to_nifti("dicom/case_01/", "out/")
    """
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: '{dicom_dir}'")
    ensure_dir(output_path)

    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not file_names:
        raise FileNotFoundError(f"No DICOM series found in '{dicom_dir}'.")
    reader.SetFileNames(file_names)
    image = reader.Execute()

    name = os.path.basename(os.path.normpath(dicom_dir)) or "series"
    out_file = os.path.join(output_path, f"{name}.nii.gz")
    sitk.WriteImage(image, out_file)

    if debug:
        logger.info("NIfTI written from DICOM series at: '%s' (%d slices)",
                    out_file, len(file_names))
    return out_file


def nifti_to_dicom(nii_path: str,
                   output_path: str,
                   series_description: str = "nidataset",
                   debug: bool = False) -> str:
    """
    Convert a NIfTI volume into a DICOM series (one ``.dcm`` file per slice).

    Geometry (spacing, origin, orientation) is preserved and float data is cast
    to ``int16`` (standard for CT/MR storage). Slices share a generated
    SeriesInstanceUID and carry per-slice position/instance tags so viewers load
    them as one volume.

    Writes ``<PREFIX>/0000.dcm ...`` under ``output_path``.

    :param nii_path:           Path to the input NIfTI file.
    :param output_path:        Directory under which the series folder is created.
    :param series_description: DICOM SeriesDescription tag (0008,103E).
    :param debug:              If ``True``, logs the output directory.
    :returns: Path to the created DICOM series directory.

    Example
    -------
    >>> from nidataset.transforms import nifti_to_dicom
    >>> nifti_to_dicom("scan.nii.gz", "out/")
    """
    import time

    validate_nifti_path(nii_path)

    image = sitk.ReadImage(nii_path)
    if image.GetPixelID() not in (sitk.sitkInt16, sitk.sitkUInt16):
        image = sitk.Cast(image, sitk.sitkInt16)

    prefix = strip_nifti_ext(os.path.basename(nii_path))
    series_dir = os.path.join(output_path, prefix)
    ensure_dir(series_dir)

    # Shared UID root + tags applied to every slice.
    now = time.strftime("%Y%m%d")
    time_str = time.strftime("%H%M%S")
    series_uid = "1.2.826.0.1.3680043.2.1125." + time_str + ".1." + str(int(time.time()))
    direction = image.GetDirection()
    shared_tags = {
        "0008|0060": "OT",                                   # Modality (Other)
        "0008|0020": now,                                    # StudyDate
        "0008|0030": time_str,                               # StudyTime
        "0008|103e": series_description,                     # SeriesDescription
        "0020|000e": series_uid,                             # SeriesInstanceUID
        "0020|0037": "\\".join(str(direction[i]) for i in (0, 3, 6, 1, 4, 7)),  # Orientation
    }

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    depth = image.GetDepth()
    for z in range(depth):
        slice_img = image[:, :, z]
        for tag, value in shared_tags.items():
            slice_img.SetMetaData(tag, value)
        position = image.TransformIndexToPhysicalPoint((0, 0, z))
        slice_img.SetMetaData("0020|0032", "\\".join(str(p) for p in position))  # ImagePosition
        slice_img.SetMetaData("0020|0013", str(z + 1))                            # InstanceNumber
        slice_img.SetMetaData("0008|0018", series_uid + "." + str(z + 1))         # SOPInstanceUID
        writer.SetFileName(os.path.join(series_dir, f"{z:04d}.dcm"))
        writer.Execute(slice_img)

    if debug:
        logger.info("DICOM series written at: '%s' (%d slices)", series_dir, depth)
    return series_dir
