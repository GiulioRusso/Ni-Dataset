"""
Shared helpers for nidataset modules.

Provides common validation, file discovery, and directory creation utilities
to eliminate duplicated boilerplate across the package.
"""

import json
import logging
import os
from collections.abc import Generator
from typing import Callable

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
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' or '.nii' file. Got '{nii_path}' instead.")


def list_nifti_files(folder: str) -> list[str]:
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
) -> Generator[tuple[str, str], None, None]:
    """
    Yield ``(full_path, prefix)`` for every NIfTI file in *folder* (sorted).

    :raises FileNotFoundError: If the folder does not exist or has no NIfTI files.
    """
    for fname in list_nifti_files(folder):
        yield os.path.join(folder, fname), strip_nifti_ext(fname)


def run_nifti_dataset(
    per_file: Callable[..., str],
    nii_folder: str,
    output_path: str,
    desc: str,
    debug: bool = False,
    **kwargs,
) -> list[str]:
    """
    Apply *per_file* to every NIfTI in *nii_folder* with a tqdm progress bar.

    *per_file* must accept ``(nii_path, output_path, **kwargs, debug=...)`` and
    return the output path. Failures are logged and skipped, matching the
    per-module ``*_dataset`` idiom.

    :returns: List of output paths for the files that succeeded.
    """
    from tqdm import tqdm  # local import: keeps import-time deps minimal

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)
    results: list[str] = []
    for nii_file in tqdm(nii_files, desc=desc, unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)
        try:
            results.append(per_file(nii_path, output_path, debug=debug, **kwargs))
        except Exception as e:
            logger.warning("Error processing %s: %s", nii_file, e)
    return results


def validate_view(view: str) -> None:
    """
    Validate an anatomical view string.

    :raises ValueError: If view is not 'axial', 'coronal', or 'sagittal'.
    """
    valid_views = {"axial", "coronal", "sagittal"}
    if view not in valid_views:
        raise ValueError(f"Error: view must be one of {valid_views}. Got '{view}' instead.")


def load_mapping(path: str) -> dict:
    """
    Load a ``.json`` (stdlib) or ``.yaml`` / ``.yml`` (needs PyYAML) file whose root is a mapping.

    An empty file loads as ``{}``.

    :raises FileNotFoundError: If *path* does not exist.
    :raises ImportError:       If a YAML file is given but PyYAML is missing.
    :raises ValueError:        On an unsupported extension or a non-mapping root.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: '{path}'")
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # ponytail: yaml is optional, JSON works dep-free
            raise ImportError(
                "YAML config requires the optional 'pyyaml' package. "
                "Install it (`pip install 'nidataset[yaml]'`) or use a .json config instead."
            ) from exc
        data = yaml.safe_load(text) or {}
    elif ext == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        raise ValueError(f"Unsupported config extension '{ext}'. Use .json, .yaml or .yml.")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
    return data
