"""
Analysis routines for NIfTI datasets.

Provides volume comparison metrics, intensity statistics, and dataset splitting.
"""

import os
import csv
import json
import random
import logging
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

from ._helpers import (
    validate_nifti_path,
    list_nifti_files,
    ensure_dir,
    strip_nifti_ext,
    is_nifti,
)

logger = logging.getLogger("nidataset")


# ---------------------------------------------------------------------------
# Volume comparison
# ---------------------------------------------------------------------------


def compare_volumes(nii_path_a: str,
                    nii_path_b: str,
                    metrics: Optional[List[str]] = None,
                    mask_path: Optional[str] = None,
                    debug: bool = False) -> Dict[str, float]:
    """
    Compute voxel-wise similarity metrics between two NIfTI volumes.

    Useful for evaluating segmentation quality, registration accuracy, or
    comparing annotations. Both volumes must have the same shape.

    Supported metrics:
        - ``"dice"``      – Dice similarity coefficient (for binary volumes)
        - ``"jaccard"``   – Jaccard index / IoU (for binary volumes)
        - ``"hausdorff"`` – Hausdorff distance in voxels (for binary volumes)
        - ``"mse"``       – Mean Squared Error
        - ``"mae"``       – Mean Absolute Error
        - ``"psnr"``      – Peak Signal-to-Noise Ratio
        - ``"volume_diff"`` – Absolute difference in non-zero voxel count
        - ``"correlation"`` – Pearson correlation coefficient

    :param nii_path_a:
        Path to the first NIfTI file.

    :param nii_path_b:
        Path to the second NIfTI file.

    :param metrics:
        List of metric names to compute. If ``None``, computes all available
        metrics.

    :param mask_path:
        Optional path to a mask NIfTI file. If provided, metrics are computed
        only within the masked region (non-zero voxels of the mask).

    :param debug:
        If ``True``, logs detailed information.

    :returns:
        Dictionary mapping metric names to their computed values.

    :raises FileNotFoundError:
        If any input file does not exist.

    :raises ValueError:
        If volumes have different shapes or unknown metrics are requested.

    Example
    -------
    >>> from nidataset.analysis import compare_volumes
    >>>
    >>> results = compare_volumes(
    ...     "segmentation_pred.nii.gz",
    ...     "segmentation_gt.nii.gz",
    ...     metrics=["dice", "hausdorff"],
    ... )
    >>> print(results)
    {'dice': 0.87, 'hausdorff': 4.2}
    """

    all_metrics = {"dice", "jaccard", "hausdorff", "mse", "mae", "psnr",
                   "volume_diff", "correlation"}

    if metrics is None:
        metrics = sorted(all_metrics)
    else:
        unknown = set(metrics) - all_metrics
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Available: {sorted(all_metrics)}")

    validate_nifti_path(nii_path_a)
    validate_nifti_path(nii_path_b)

    data_a = nib.load(nii_path_a).get_fdata()
    data_b = nib.load(nii_path_b).get_fdata()

    if data_a.shape != data_b.shape:
        raise ValueError(
            f"Shape mismatch: '{nii_path_a}' has shape {data_a.shape}, "
            f"'{nii_path_b}' has shape {data_b.shape}."
        )

    if mask_path is not None:
        validate_nifti_path(mask_path)
        mask = nib.load(mask_path).get_fdata() > 0
        data_a = data_a[mask]
        data_b = data_b[mask]

    results: Dict[str, float] = {}

    # Binary metrics
    bin_a = (data_a > 0).astype(np.float64)
    bin_b = (data_b > 0).astype(np.float64)
    intersection = np.sum(bin_a * bin_b)
    sum_ab = np.sum(bin_a) + np.sum(bin_b)

    if "dice" in metrics:
        results["dice"] = (2.0 * intersection / sum_ab) if sum_ab > 0 else 1.0

    if "jaccard" in metrics:
        union = sum_ab - intersection
        results["jaccard"] = (intersection / union) if union > 0 else 1.0

    if "hausdorff" in metrics:
        if mask_path is not None:
            logger.warning("Hausdorff distance is computed on the full volume, ignoring mask.")
        # Reload full volumes for distance computation
        full_a = nib.load(nii_path_a).get_fdata() > 0
        full_b = nib.load(nii_path_b).get_fdata() > 0
        if not np.any(full_a) or not np.any(full_b):
            results["hausdorff"] = float("inf")
        else:
            dist_a = distance_transform_edt(~full_a)
            dist_b = distance_transform_edt(~full_b)
            results["hausdorff"] = max(np.max(dist_a[full_b]), np.max(dist_b[full_a]))

    # Intensity metrics
    flat_a = data_a.ravel().astype(np.float64)
    flat_b = data_b.ravel().astype(np.float64)

    if "mse" in metrics:
        results["mse"] = float(np.mean((flat_a - flat_b) ** 2))

    if "mae" in metrics:
        results["mae"] = float(np.mean(np.abs(flat_a - flat_b)))

    if "psnr" in metrics:
        mse = np.mean((flat_a - flat_b) ** 2)
        max_val = max(np.max(np.abs(flat_a)), np.max(np.abs(flat_b)))
        if mse == 0:
            results["psnr"] = float("inf")
        elif max_val == 0:
            results["psnr"] = 0.0
        else:
            results["psnr"] = float(10.0 * np.log10(max_val ** 2 / mse))

    if "volume_diff" in metrics:
        results["volume_diff"] = abs(int(np.count_nonzero(data_a)) - int(np.count_nonzero(data_b)))

    if "correlation" in metrics:
        if np.std(flat_a) == 0 or np.std(flat_b) == 0:
            results["correlation"] = 0.0
        else:
            results["correlation"] = float(np.corrcoef(flat_a, flat_b)[0, 1])

    if debug:
        for k, v in results.items():
            logger.info("  %s: %.6f", k, v)

    return results


def compare_volumes_dataset(nii_folder_a: str,
                            nii_folder_b: str,
                            output_path: str,
                            metrics: Optional[List[str]] = None,
                            debug: bool = False) -> str:
    """
    Compute comparison metrics between matching NIfTI files in two folders.

    Files are matched by filename. Results are saved as
    ``volume_comparison.csv``.

    :param nii_folder_a: First folder of NIfTI files.
    :param nii_folder_b: Second folder of NIfTI files.
    :param output_path:  Directory for the output CSV.
    :param metrics:      List of metric names (see ``compare_volumes``).
    :param debug:        If ``True``, logs details.

    :returns: Path to the saved CSV file.

    Example
    -------
    >>> compare_volumes_dataset("pred/", "gt/", "output/", metrics=["dice", "mse"])
    """

    files_a = set(list_nifti_files(nii_folder_a))
    files_b = set(list_nifti_files(nii_folder_b))
    common = sorted(files_a & files_b)

    if not common:
        raise FileNotFoundError("No matching NIfTI filenames found between the two folders.")

    ensure_dir(output_path)

    if metrics is None:
        metrics = sorted({"dice", "jaccard", "hausdorff", "mse", "mae", "psnr",
                          "volume_diff", "correlation"})

    rows = []
    for fname in tqdm(common, desc="Comparing volumes", unit="file"):
        path_a = os.path.join(nii_folder_a, fname)
        path_b = os.path.join(nii_folder_b, fname)
        try:
            result = compare_volumes(path_a, path_b, metrics=metrics, debug=debug)
            rows.append([fname] + [result.get(m, "") for m in metrics])
        except Exception as e:
            logger.warning("Error comparing %s: %s", fname, e)

    csv_path = os.path.join(output_path, "volume_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FILENAME"] + [m.upper() for m in metrics])
        writer.writerows(rows)

    logger.info("Comparison results saved in: '%s'", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Intensity statistics
# ---------------------------------------------------------------------------


def compute_volume_statistics(nii_path: str,
                              mask_path: Optional[str] = None,
                              debug: bool = False) -> Dict[str, float]:
    """
    Compute detailed intensity statistics for a single NIfTI volume.

    Statistics include: mean, std, min, max, median, percentiles (1, 5, 25,
    75, 95, 99), skewness, kurtosis, nonzero voxel count, and SNR within
    the optionally masked region.

    :param nii_path:  Path to the NIfTI file.
    :param mask_path: Optional mask; stats are computed within mask only.
    :param debug:     If ``True``, logs details.

    :returns: Dictionary of statistic names to values.

    Example
    -------
    >>> stats = compute_volume_statistics("scan.nii.gz")
    >>> print(stats["mean"], stats["std"])
    """

    validate_nifti_path(nii_path)
    data = nib.load(nii_path).get_fdata()

    if mask_path is not None:
        validate_nifti_path(mask_path)
        mask = nib.load(mask_path).get_fdata() > 0
        values = data[mask].astype(np.float64)
    else:
        values = data.ravel().astype(np.float64)

    nonzero_values = values[values != 0]

    stats: Dict[str, float] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "percentile_1": float(np.percentile(values, 1)),
        "percentile_5": float(np.percentile(values, 5)),
        "percentile_25": float(np.percentile(values, 25)),
        "percentile_75": float(np.percentile(values, 75)),
        "percentile_95": float(np.percentile(values, 95)),
        "percentile_99": float(np.percentile(values, 99)),
        "nonzero_count": int(np.count_nonzero(values)),
        "total_voxels": int(values.size),
    }

    # Skewness
    if stats["std"] > 0:
        stats["skewness"] = float(np.mean(((values - stats["mean"]) / stats["std"]) ** 3))
        stats["kurtosis"] = float(np.mean(((values - stats["mean"]) / stats["std"]) ** 4) - 3.0)
    else:
        stats["skewness"] = 0.0
        stats["kurtosis"] = 0.0

    # SNR (mean / std of nonzero region)
    if len(nonzero_values) > 0 and np.std(nonzero_values) > 0:
        stats["snr"] = float(np.mean(nonzero_values) / np.std(nonzero_values))
    else:
        stats["snr"] = 0.0

    if debug:
        for k, v in stats.items():
            logger.info("  %s: %s", k, v)

    return stats


def compute_volume_statistics_dataset(nii_folder: str,
                                      output_path: str,
                                      mask_folder: Optional[str] = None,
                                      debug: bool = False) -> str:
    """
    Compute intensity statistics for all NIfTI files in a folder and save
    as ``volume_statistics.csv``.

    :param nii_folder:  Folder containing NIfTI files.
    :param output_path: Directory for the output CSV.
    :param mask_folder: Optional folder with matching mask files.
    :param debug:       If ``True``, logs details.

    :returns: Path to the saved CSV file.

    Example
    -------
    >>> compute_volume_statistics_dataset("scans/", "output/")
    """

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)

    all_rows = []
    header = None

    for nii_file in tqdm(nii_files, desc="Computing statistics", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)
        mask_path = None
        if mask_folder is not None:
            candidate = os.path.join(mask_folder, nii_file)
            if os.path.isfile(candidate):
                mask_path = candidate

        try:
            stats = compute_volume_statistics(nii_path, mask_path=mask_path, debug=debug)
            if header is None:
                header = sorted(stats.keys())
            all_rows.append([nii_file] + [stats[k] for k in header])
        except Exception as e:
            logger.warning("Error computing stats for %s: %s", nii_file, e)

    csv_path = os.path.join(output_path, "volume_statistics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FILENAME"] + [h.upper() for h in (header or [])])
        writer.writerows(all_rows)

    logger.info("Volume statistics saved in: '%s'", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def split_dataset(nii_folder: str,
                  output_path: str,
                  ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                  seed: int = 42,
                  copy_files: bool = False,
                  debug: bool = False) -> Dict[str, List[str]]:
    """
    Split a NIfTI dataset folder into train / val / test subsets.

    Creates a JSON manifest (``split.json``) listing which files belong to
    each subset, and optionally copies (or symlinks) files into
    ``train/``, ``val/``, ``test/`` subdirectories.

    :param nii_folder:  Folder containing NIfTI files.
    :param output_path: Directory where split artifacts are saved.
    :param ratios:      ``(train, val, test)`` fractions summing to 1.0.
    :param seed:        Random seed for reproducibility.
    :param copy_files:  If ``True``, copies files into subset folders.
                        If ``False`` (default), only writes the manifest.
    :param debug:       If ``True``, logs split counts.

    :returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        lists of filenames.

    :raises ValueError: If ratios do not sum to approximately 1.0.

    Example
    -------
    >>> splits = split_dataset("dataset/scans/", "output/", ratios=(0.8, 0.1, 0.1))
    >>> print(len(splits["train"]), len(splits["val"]), len(splits["test"]))
    """

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0. Got {sum(ratios):.6f}.")
    if len(ratios) != 3:
        raise ValueError("Ratios must be a tuple of 3 values (train, val, test).")

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)

    rng = random.Random(seed)
    shuffled = list(nii_files)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits: Dict[str, List[str]] = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    # Save manifest
    manifest_path = os.path.join(output_path, "split.json")
    with open(manifest_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info("Split manifest saved: '%s'", manifest_path)

    # Optionally copy files
    if copy_files:
        import shutil
        for subset, files in splits.items():
            subset_dir = os.path.join(output_path, subset)
            ensure_dir(subset_dir)
            for fname in files:
                src = os.path.join(nii_folder, fname)
                dst = os.path.join(subset_dir, fname)
                shutil.copy2(src, dst)
        logger.info("Files copied into train/val/test subdirectories.")

    if debug:
        for subset, files in splits.items():
            logger.info("  %s: %d files", subset, len(files))

    return splits
