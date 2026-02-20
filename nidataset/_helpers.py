"""
Shared helpers for nidataset modules.

Provides common validation, file discovery, and directory creation utilities
to eliminate duplicated boilerplate across the package.
"""

import os
import logging
from typing import List, Generator, Tuple

logger = logging.getLogger("nidataset")

# Valid NIfTI extensions
NIFTI_EXTENSIONS = (".nii.gz", ".nii")


def is_nifti(path: str) -> bool:
    """Check whether a file path has a valid NIfTI extension."""
    return path.endswith(".nii.gz") or path.endswith(".nii")


def strip_nifti_ext(filename: str) -> str:
    """Remove the NIfTI extension (.nii.gz or .nii) from a filename."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def validate_nifti_path(nii_path: str) -> None:
    """
    Validate that a NIfTI file exists and has a valid extension.

    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file does not have a .nii.gz or .nii extension.
    """
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")
    if not is_nifti(nii_path):
        raise ValueError(
            f"Error: invalid file format. Expected a '.nii.gz' or '.nii' file. Got '{nii_path}' instead."
        )


def list_nifti_files(folder: str) -> List[str]:
    """
    Return a sorted list of NIfTI filenames inside *folder*.

    :raises FileNotFoundError: If the folder does not exist or contains no NIfTI files.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Error: the folder '{folder}' does not exist.")
    nii_files = sorted(f for f in os.listdir(folder) if is_nifti(f))
    if not nii_files:
        raise FileNotFoundError(f"Error: no NIfTI files found in '{folder}'.")
    return nii_files


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def iterate_nifti_dataset(
    folder: str,
) -> Generator[Tuple[str, str], None, None]:
    """
    Yield ``(full_path, prefix)`` for every NIfTI file in *folder* (sorted).

    :raises FileNotFoundError: If the folder does not exist or has no NIfTI files.
    """
    for fname in list_nifti_files(folder):
        yield os.path.join(folder, fname), strip_nifti_ext(fname)


def validate_view(view: str) -> None:
    """
    Validate an anatomical view string.

    :raises ValueError: If view is not 'axial', 'coronal', or 'sagittal'.
    """
    valid_views = {"axial", "coronal", "sagittal"}
    if view not in valid_views:
        raise ValueError(f"Error: view must be one of {valid_views}. Got '{view}' instead.")
