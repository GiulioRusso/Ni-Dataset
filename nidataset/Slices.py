import os
from tqdm import tqdm
import numpy as np
import csv
import nibabel as nib
from PIL import Image


def extract_slices(nii_path: str, 
                   output_path: str, 
                   view: str = "axial", 
                   debug: bool = False) -> None:
    """
    Extracts slices from a NIfTI file and saves them as images .tif, based on the selected view and named with the NIfTI filename with the progressive slice number.

    :param nii_path: path to the input .nii.gz file with shape (X, Y, Z).
    :param output_path: path where the extracted slices will be saved.
    :param view: "axial" -> extracts along the Z-axis.
                 "coronal" -> extracts along the Y-axis.
                 "sagittal" -> extracts along the X-axis.
    :param debug: verbose print about the total number of slices extracted. default is False.

    :raises FileNotFoundError: if the input NIfTI file does not exist.
    :raises ValueError: if the NIfTI file is empty, has invalid extension, or an invalid view is provided.
    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. expected a '.nii.gz' file, but got '{nii_path}'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the NIfTI file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()

    # validate NIfTI data dimensions
    if nii_data.ndim != 3:
        raise ValueError(f"Error: expected a 3D NIfTI file, but got shape {nii_data.shape}")

    # mapping of views to slicing axes
    view_mapping = {
        "axial": (2, lambda data, i: data[:, :, i]),       # Z-axis
        "coronal": (1, lambda data, i: data[:, i, :]),     # Y-axis
        "sagittal": (0, lambda data, i: data[i, :, :])     # X-axis
    }

    # validate view
    if view not in view_mapping:
        raise ValueError("Error: view must be 'axial', 'coronal', or 'sagittal'.")

    # get axis and slicing function
    axis, slice_func = view_mapping[view]

    # get number of slices along the selected axis
    num_slices = nii_data.shape[axis]
    
    # check if the dimension is not zero    
    if num_slices == 0:
        raise ValueError("Error: the NIfTI file contains no slices (empty volume).")

    # define prefix as the nii.gz filename
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")

    # iterate over slices and save as images
    for i in tqdm(range(num_slices), desc=f"Processing {prefix} ({view})", unit="slice"):
        # extract the slice using the dynamic function
        slice_data = slice_func(nii_data, i)

        # construct filename with zero-padded slice index
        slice_filename = f"{prefix}_{str(i).zfill(3)}.tif"
        slice_path = os.path.join(output_path, slice_filename)

        # save slice as an image
        slice_to_save = Image.fromarray(slice_data)
        slice_to_save.save(slice_path)

    # debug verbose print
    if debug:
        print(f"\nInput file: '{nii_path}'\nOutput path: '{output_path}'\nTotal {view} slices extracted: {num_slices}")


def extract_slices_from_dataset(nii_folder: str, 
                                output_path: str, 
                                view: str = "axial", 
                                saving_mode: str = "case", 
                                save_stats: bool = False) -> None:
    """
    Extracts slices from all NIfTI files in a dataset folder and saves them as images .tif, based on the selected view and named with the NIfTI filename with the progressive slice number.

    :param nii_folder: path to the folder containing all .nii.gz files with shape (X, Y, Z).
    :param output_path: path where the extracted slices will be saved.
    :param view: "axial" -> extracts along the Z-axis.
                 "coronal" -> extracts along the Y-axis.
                 "sagittal" -> extracts along the X-axis.
    :param saving_mode: "case" -> creates a folder for each case.
                        "view" -> saves all slices inside a single view folder.
    :param save_stats: if True, saves a CSV file with FILENAME and NUM_SLICES information per case.

    :raises FileNotFoundError: if the dataset folder does not exist or contains no .nii.gz files.
    :raises ValueError: if an invalid view or saving_mode is provided.
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{nii_folder}' does not exist.")

    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{nii_folder}'.")

    # validate input parameters
    if saving_mode not in ["case", "view"]:
        raise ValueError("error: saving_mode must be either 'case' or 'view'.")
    if view not in ["axial", "coronal", "sagittal"]:
        raise ValueError("error: view must be 'axial', 'coronal', or 'sagittal'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create a single folder if using "view" mode
    if saving_mode == "view":
        view_output_dir = os.path.join(output_path, view)
        os.makedirs(view_output_dir, exist_ok=True)

    # initialize statistics tracking
    stats = []
    total_slices = 0
    stats_file = os.path.join(output_path, f"{view}_slices_stats.csv") if save_stats else None

    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc=f"Processing {view} slices", unit="file"):
        # nii.gz file path
        nii_path = os.path.join(nii_folder, nii_file)

        # extract the filename prefix (case ID)
        prefix = os.path.basename(nii_path).replace(".nii.gz", "")

        # update tqdm description with the current file prefix
        tqdm.write(f"Processing: {prefix}")

        # determine number of slices **before** calling extract_slices
        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
            num_slices = nii_data.shape[{"axial": 2, "coronal": 1, "sagittal": 0}[view]]
        except Exception as e:
            tqdm.write(f"Error processing {nii_file} for statistical analysis: {e}")
            continue  # skip this file if an error occurs
        
        # keep track of the total number of slices
        total_slices += num_slices
        if save_stats:
            stats.append([nii_file, num_slices])

        # determine the appropriate output folder
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix, view)
            os.makedirs(case_output_dir, exist_ok=True)
            extract_slices(nii_path, case_output_dir, view, debug=False)
        else:
            extract_slices(nii_path, view_output_dir, view, debug=False)

    # save statistics if enabled
    if save_stats:
        with open(stats_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["FILENAME", "NUM_SLICES"])
            writer.writerows(stats)
            writer.writerow(["TOTAL_SLICES", total_slices])
        
        print(f"\nSlice extraction statistics saved in: '{stats_file}'")


def extract_annotations(nii_path: str, 
                        output_path: str, 
                        view: str = "axial",
                        saving_mode: str = "slice", 
                        data_mode: str = "center",
                        debug: bool = False) -> None:
    """
    Extracts annotations from a NIfTI  annotation file and saves them as CSV, based on the selected view and named with the NIfTI filename with the progressive slice number.

    :param nii_path: path to the input .nii.gz file with shape (X, Y, Z).
    :param output_path: path where the CSV annotations will be saved.
    :param view: "axial" -> extracts along the Z-axis.
                 "coronal" -> extracts along the Y-axis.
                 "sagittal" -> extracts along the X-axis.
    :param saving_mode: "slice" -> generates a CSV per slice.
                        "volume" -> generates a single CSV for the whole volume.
    :param data_mode: "center" -> saves the center (X, Y, Z) of the bounding box.
                      "box" -> saves the bounding box coordinates.
    :param debug: if True, prints additional information about the extraction.

    :raises FileNotFoundError: if the annotation file does not exist.
    :raises ValueError: if invalid view, saving_mode, or data_mode is provided.
    """

    # check if the input annotation file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"error: the annotation file '{nii_path}' does not exist.")

    # validate input parameters
    if view not in ["axial", "coronal", "sagittal"]:
        raise ValueError("error: view must be 'axial', 'coronal', or 'sagittal'.")
    if saving_mode not in ["slice", "volume"]:
        raise ValueError("error: saving_mode must be either 'slice' or 'volume'.")
    if data_mode not in ["center", "box"]:
        raise ValueError("error: data_mode must be either 'center' or 'box'.")

    # load the 3D annotation file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()

    # extract filename prefix
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # identify all 3D bounding boxes
    unique_labels = np.unique(nii_data)
    unique_labels = unique_labels[unique_labels > 0]

    bounding_boxes = []

    # iterate over unique labels to extract bounding box information
    for label in tqdm(unique_labels, desc=f"Processing {prefix}", unit="box"):
        positions = np.argwhere(nii_data == label)

        # get min/max for bounding box
        min_x, min_y, min_z = np.min(positions, axis=0)
        max_x, max_y, max_z = np.max(positions, axis=0)

        # adjust coordinates based on view
        if view == "axial":
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            box_data = [center_x, center_y, min_z, max_z] if data_mode == "center" else [min_x, min_y, min_z, max_x, max_y, max_z]
        elif view == "coronal":
            center_x, center_z = (min_x + max_x) / 2, (min_z + max_z) / 2
            box_data = [center_x, center_z, min_y, max_y] if data_mode == "center" else [min_x, min_z, min_y, max_x, max_z, max_y]
        else:  # sagittal
            center_y, center_z = (min_y + max_y) / 2, (min_z + max_z) / 2
            box_data = [center_y, center_z, min_x, max_x] if data_mode == "center" else [min_y, min_z, min_x, max_y, max_z, max_x]

        bounding_boxes.append(box_data)

    # handle extraction mode
    if saving_mode == "volume":
        csv_file = os.path.join(output_path, f"{prefix}.csv")
        with open(csv_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # write appropriate headers based on data_mode
            if data_mode == "center":
                writer.writerow(["CENTER_X", "CENTER_Y", "CENTER_Z"])
                bounding_boxes_vol = [[box[0], box[1], box[2]] for box in bounding_boxes]  # Remove min/max range
            else:
                writer.writerow(["X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"])
                bounding_boxes_vol = bounding_boxes  # Keep full bounding box

            writer.writerows(bounding_boxes_vol)

        # debug print
        if debug:
            print(f"\nInput file: '{nii_path}'\nOutput path: '{output_path}'\nTotal volume annotations extracted: {len(bounding_boxes)}")

    else:
        slice_annotations = {}

        # build a dict record for each slice with annotation
        for box in bounding_boxes:
            z_min, z_max = int(box[2]), int(box[3] if data_mode == "center" else box[5])

            for z in range(z_min, z_max + 1):
                if z not in slice_annotations:
                    slice_annotations[z] = []

                if data_mode == "center":
                    # store only X and Y for center mode
                    slice_annotations[z].append([int(box[0]), int(box[1])])
                else:
                    # store full bounding box for box mode
                    slice_annotations[z].append([int(box[0]), int(box[1]), int(box[3]), int(box[4])])

        # process each slice
        num_slices = len(slice_annotations)
        for z, boxes in tqdm(slice_annotations.items(), desc=f"Processing {prefix} (Slices)", unit="slice"):
            slice_filename = f"{prefix}_{str(z).zfill(3)}.csv"
            slice_file = os.path.join(output_path, slice_filename)

            with open(slice_file, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if data_mode == "center":
                    writer.writerow(["CENTER_X", "CENTER_Y"])
                else:
                    writer.writerow(["X_MIN", "Y_MIN", "X_MAX", "Y_MAX"])
                writer.writerows(boxes)

        # debug print
        if debug:
            print(f"\nInput file: '{nii_path}'\nOutput path: '{output_path}'\nTotal slices with annotations extracted: {num_slices}")


def extract_annotations_from_dataset(nii_folder: str, 
                                     output_path: str, 
                                     view: str = "axial",
                                     saving_mode: str = "case", 
                                     extraction_mode: str = "slice", 
                                     data_mode: str = "center",
                                     save_stats: bool = False) -> None:
    """
    Extracts annotations from all NIfTI annotation files in a dataset folder and saves them as CSV.

    :param nii_folder: path to the folder containing all .nii.gz files with shape (X, Y, Z).
    :param output_path: path where the extracted annotations will be saved.
    :param view: "axial" -> extracts along the Z-axis.
                 "coronal" -> extracts along the Y-axis.
                 "sagittal" -> extracts along the X-axis.
    :param saving_mode: "case" -> creates a folder for each case.
                        "view" -> saves all CSVs inside a single folder.
    :param extraction_mode: "slice" -> generates a CSV per slice.
                            "volume" -> generates a single CSV for the whole volume.
    :param data_mode: "center" -> saves the center (X, Y, Z) of the bounding box.
                      "box" -> saves the bounding box coordinates.
    :param save_stats: if True, saves a CSV file with FILENAME and NUM_ANNOTATIONS information per case.

    :raises FileNotFoundError: if the dataset folder does not exist or contains no .nii.gz files.
    :raises ValueError: if invalid view, saving_mode, extraction_mode, or data_mode is provided.
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"error: the dataset folder '{nii_folder}' does not exist.")

    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"error: no .nii.gz files found in '{nii_folder}'.")

    # validate input parameters
    if view not in ["axial", "coronal", "sagittal"]:
        raise ValueError("error: view must be 'axial', 'coronal', or 'sagittal'.")
    if saving_mode not in ["case", "view"]:
        raise ValueError("error: saving_mode must be either 'case' or 'view'.")
    if extraction_mode not in ["slice", "volume"]:
        raise ValueError("error: extraction_mode must be either 'slice' or 'volume'.")
    if data_mode not in ["center", "box"]:
        raise ValueError("error: data_mode must be either 'center' or 'box'.")
    
    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # create a single folder for the chosen view if using "view" mode
    if saving_mode == "view":
        view_output_dir = os.path.join(output_path, view)
        os.makedirs(view_output_dir, exist_ok=True)

    # initialize statistics tracking
    stats = []
    total_annotations = 0
    stats_file = os.path.join(output_path, f"{view}_annotations_stats.csv") if save_stats else None

    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Processing NIfTI files", unit="file"):
        # nii.gz file path
        nii_path = os.path.join(nii_folder, nii_file)

        # extract the filename prefix (case ID)
        prefix = os.path.basename(nii_path).replace(".nii.gz", "")

        # update tqdm description with the current file prefix
        tqdm.write(f"Processing: {prefix}")

        # determine the number of annotations **before** calling extract_annotations
        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
            unique_labels = np.unique(nii_data)
            num_annotations = len(unique_labels[unique_labels > 0])  # Count non-zero annotations
        except Exception as e:
            tqdm.write(f"error processing {nii_file} for statistical analysis: {e}")
            continue  # skip this file if an error occurs
        
        # keep track of the total number of annotations
        total_annotations += num_annotations
        if save_stats:
            stats.append([nii_file, num_annotations])

        # determine the appropriate output folder
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix, view)
            os.makedirs(case_output_dir, exist_ok=True)
            extract_annotations(nii_path, case_output_dir, view, extraction_mode, data_mode, debug=False)
        else:
            extract_annotations(nii_path, view_output_dir, view, extraction_mode, data_mode, debug=False)

    # save statistics if enabled
    if save_stats:
        with open(stats_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["FILENAME", "NUM_ANNOTATIONS"])
            writer.writerows(stats)
            writer.writerow(["TOTAL_ANNOTATIONS", total_annotations])
        
        print(f"\nAnnotation statistics saved in: '{stats_file}'")

