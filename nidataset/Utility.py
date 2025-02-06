import os
import csv
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label
import scipy.ndimage as ndi


def dataset_images_info(nii_folder: str,
                        output_path: str) -> None:
    """
    Extracts metadata from all NIfTI files in a dataset and saves the results in a CSV file.

    :param nii_folder: Path to the folder containing .nii.gz files.
    :param output_path: Path where the metadata CSV file will be saved.

    :raises FileNotFoundError: If the dataset folder does not exist or contains no .nii.gz files.
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: The dataset folder '{nii_folder}' does not exist.")

    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: No .nii.gz files found in '{nii_folder}'.")
    
    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # list to store extracted metadata
    metadata = []

    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Extracting metadata", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)

        try:
            # load the NIfTI file
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
            affine = nii_img.affine
            header = nii_img.header

            # extract metadata
            shape = nii_data.shape  # image dimensions (X, Y, Z)
            voxel_size = header.get_zooms()[:3]  # voxel size in mm
            dtype = nii_data.dtype  # data type
            min_intensity = np.min(nii_data)  # min voxel intensity
            max_intensity = np.max(nii_data)  # max voxel intensity

            # calculate the volume of nonzero (brain) pixels
            brain_voxel_count = np.count_nonzero(nii_data)
            brain_volume = brain_voxel_count * np.prod(voxel_size)  # in mm³

            # find the bounding box containing nonzero voxels (brain region)
            nonzero_coords = np.array(np.nonzero(nii_data))
            min_coords = nonzero_coords.min(axis=1)
            max_coords = nonzero_coords.max(axis=1)

            # add to metadata list
            metadata.append([
                nii_file, shape, voxel_size, dtype, min_intensity, max_intensity, 
                brain_voxel_count, brain_volume, min_coords.tolist(), max_coords.tolist()
            ])

        except Exception as e:
            print(f"Error processing {nii_file}: {e}")
            continue  # skip this file if an error occurs
    
    
    output_csv = os.path.join(output_path, "dataset_images_info.csv")

    # write metadata to CSV file
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "FILENAME", "SHAPE (X, Y, Z)", "VOXEL SIZE (mm)", "DATA TYPE", "MIN VALUE", "MAX VALUE",
            "BRAIN VOXELS", "BRAIN VOLUME (mm³)", "BBOX MIN (X, Y, Z)", "BBOX MAX (X, Y, Z)"
        ])
        writer.writerows(metadata)

    print(f"\nDataset metadata saved in: '{output_csv}'")


def dataset_annotations_info(nii_folder: str,
                             output_path: str, 
                             annotation_value: int = 1) -> None:
    """
    Extracts 3D bounding boxes from all NIfTI annotation files in a dataset and saves the results in a CSV file.

    :param nii_folder: Path to the folder containing .nii.gz annotation files.
    :param output_path: Path where the bounding box CSV file will be saved.
    :param annotation_value: Value in the mask representing the annotated region (default: 1).

    :raises FileNotFoundError: If the dataset folder does not exist or contains no .nii.gz files.
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: The dataset folder '{nii_folder}' does not exist.")

    # get all .nii.gz annotation files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: No .nii.gz annotation files found in '{nii_folder}'.")

    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # list to store extracted bounding box data
    bounding_boxes_info = []

    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Extracting 3D Bounding Boxes", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)

        try:
            # load the NIfTI annotation file
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()

            # filter the mask to retain only the desired annotation value
            binary_mask = (nii_data == annotation_value).astype(np.uint8)

            # find connected components in the binary mask
            labeled_components, num_components = ndi.label(binary_mask)

            # initialize bounding box storage
            bounding_boxes = []

            # iterate over detected components
            for label_idx in range(1, num_components + 1):
                # get min/max coordinates of the bounding box
                component_indices = np.argwhere(labeled_components == label_idx)
                min_coords = component_indices.min(axis=0)
                max_coords = component_indices.max(axis=0)

                # store bounding box as a list of 6 values [X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX]
                bounding_boxes.append([min_coords[0], min_coords[1], min_coords[2], 
                                       max_coords[0], max_coords[1], max_coords[2]])

            # add to bounding boxes info list
            bounding_boxes_info.append([nii_file, bounding_boxes])

        except Exception as e:
            print(f"Error processing {nii_file}: {e}")
            continue  # skip this file if an error occurs

    output_csv = os.path.join(output_path, "dataset_annotations_info.csv")

    # write bounding box data to CSV file
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["FILENAME", "3D BOXES"])
        
        for filename, boxes in bounding_boxes_info:
            writer.writerow([filename, boxes])

    print(f"\n3D bounding boxes saved in: '{output_csv}'")

