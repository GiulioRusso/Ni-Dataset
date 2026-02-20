import os
import csv
import logging
from typing import List, Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndi

from ._helpers import (
    list_nifti_files,
    ensure_dir,
    strip_nifti_ext,
)

logger = logging.getLogger("nidataset")


# Info

def dataset_images_info(nii_folder: str,
                        output_path: str) -> List[list]:
    """
    Extracts metadata from all NIfTI files in a dataset and saves it as a CSV
    named 'dataset_images_info.csv'. Extracted metadata includes image shape,
    voxel size, data type, intensity range, brain voxel count, brain volume,
    and bounding box coordinates of nonzero voxels.

    Columns:
        ["FILENAME", "SHAPE (X, Y, Z)", "VOXEL SIZE (mm)", "DATA TYPE",
         "MIN VALUE", "MAX VALUE", "BRAIN VOXELS", "BRAIN VOLUME (mm³)",
         "BBOX MIN (X, Y, Z)", "BBOX MAX (X, Y, Z)"]

    :param nii_folder: Path to the folder containing NIfTI files.
    :param output_path: Path where the metadata CSV file will be saved.

    :raises FileNotFoundError: If the dataset folder does not exist or contains no NIfTI files.

    :returns: List of metadata rows extracted from each file.

    Example:
        >>> dataset_images_info("path/to/dataset", "path/to/output_directory")
    """

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)

    metadata: List[list] = []

    for nii_file in tqdm(nii_files, desc="Extracting metadata", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)

        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
            header = nii_img.header

            shape = nii_data.shape
            voxel_size = header.get_zooms()[:3]
            dtype = nii_data.dtype
            min_intensity = np.min(nii_data)
            max_intensity = np.max(nii_data)

            brain_voxel_count = np.count_nonzero(nii_data)
            brain_volume = brain_voxel_count * np.prod(voxel_size)

            nonzero_coords = np.array(np.nonzero(nii_data))
            if nonzero_coords.size > 0:
                min_coords = nonzero_coords.min(axis=1).tolist()
                max_coords = nonzero_coords.max(axis=1).tolist()
            else:
                min_coords = [0, 0, 0]
                max_coords = [0, 0, 0]

            metadata.append([
                nii_file, shape, voxel_size, dtype, min_intensity, max_intensity,
                brain_voxel_count, brain_volume, min_coords, max_coords
            ])

        except Exception as e:
            logger.warning("Error processing %s: %s", nii_file, e)
            continue

    output_csv = os.path.join(output_path, "dataset_images_info.csv")

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "FILENAME", "SHAPE (X, Y, Z)", "VOXEL SIZE (mm)", "DATA TYPE", "MIN VALUE", "MAX VALUE",
            "BRAIN VOXELS", "BRAIN VOLUME (mm³)", "BBOX MIN (X, Y, Z)", "BBOX MAX (X, Y, Z)"
        ])
        writer.writerows(metadata)

    logger.info("Dataset metadata saved in: '%s'", output_csv)
    return metadata


def dataset_annotations_info(nii_folder: str,
                             output_path: str,
                             annotation_value: int = 1) -> List[list]:
    """
    Extracts 3D bounding boxes from all NIfTI annotation files in a dataset
    and saves them as a CSV named 'dataset_annotations_info.csv'.

    Columns:
        ["FILENAME", "3D_BOXES"]

    Each 3D box is represented as a list of 6 integers:
        [X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX]

    :param nii_folder: Path to the folder containing NIfTI annotation files.
    :param output_path: Path where the bounding box CSV file will be saved.
    :param annotation_value: Value in the mask representing the annotated region (default: 1).

    :raises FileNotFoundError: If the dataset folder does not exist or contains no NIfTI files.

    :returns: List of [filename, bounding_boxes] pairs.

    Example:
        >>> dataset_annotations_info("path/to/masks", "path/to/output_directory", annotation_value=1)
    """

    nii_files = list_nifti_files(nii_folder)
    ensure_dir(output_path)

    bounding_boxes_info: List[list] = []

    for nii_file in tqdm(nii_files, desc="Extracting 3D Bounding Boxes", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)

        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()

            binary_mask = (nii_data == annotation_value).astype(np.uint8)
            labeled_components, num_components = ndi.label(binary_mask)

            bounding_boxes = []

            for label_idx in range(1, num_components + 1):
                component_indices = np.argwhere(labeled_components == label_idx)
                min_coords = component_indices.min(axis=0)
                max_coords = component_indices.max(axis=0)

                bounding_boxes.append([min_coords[0], min_coords[1], min_coords[2],
                                       max_coords[0], max_coords[1], max_coords[2]])

            bounding_boxes_info.append([nii_file, bounding_boxes])

        except Exception as e:
            logger.warning("Error processing %s: %s", nii_file, e)
            continue

    output_csv = os.path.join(output_path, "dataset_annotations_info.csv")

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["FILENAME", "3D_BOXES"])

        for filename, boxes in bounding_boxes_info:
            writer.writerow([filename, boxes])

    logger.info("3D bounding boxes saved in: '%s'", output_csv)
    return bounding_boxes_info
