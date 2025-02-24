import os
from tqdm import tqdm
import csv
import subprocess

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from scipy.ndimage import label as ndi_label
from skimage.morphology import binary_closing, ball
from skimage.filters import threshold_otsu


def swap_nifti_views(nii_path: str, 
                     output_path: str, 
                     source_view: str, 
                     target_view: str, 
                     debug: bool = False) -> None:
    """
    Swaps anatomical views in a NIfTI image by swapping axes, applying a 90-degree rotation, and updating the affine matrix correctly.
    The output file is saved as:

        <Nifti FILENAME>_swapped_<SOURCE VIEW>_to_<TARGE VIEW>.nii.gz

    :param nii_path: path to the input NIfTI (.nii.gz) file.
    :param output_path: path where the swapped NIfTI file will be saved.
    :param source_view: current view ("axial", "coronal", or "sagittal").
    :param target_view: desired view ("axial", "coronal", or "sagittal").
    :param debug: if True, prints additional information about the transformation.

    :raises FileNotFoundError: if the input file does not exist.
    :raises ValueError: if the file is empty, has invalid dimensions, or if the views are incorrect.

    Example:

        from nidataset.Volume import swap_nifti_views

        # define paths
        nii_path = "path/to/input_image.nii.gz"
        output_path = "path/to/output_directory"

        # choose the anatomical view ('axial', 'coronal', or 'sagittal')
        source_view = "axial"
        target_view = "sagittal"

        # run the function
        swap_nifti_views(nii_path=nii_path,
                         output_path=output_path,
                         source_view=source_view,
                         target_view=target_view,
                         debug=True)

    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file. Got '{nii_path}' instead.")

    # validate the view parameters
    valid_views = {'axial', 'coronal', 'sagittal'}
    if source_view not in valid_views:
        raise ValueError(f"Error: The source view must be one of {valid_views}. Got '{source_view}' instead.")
    if target_view not in valid_views:
        raise ValueError(f"Error: The target view must be one of {valid_views}. Got '{target_view}' instead.")


    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the NIfTI file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    affine = nii_img.affine.copy()  # preserve transformation matrix
    header = nii_img.header.copy()  # preserve original header

    # validate NIfTI data dimensions
    if nii_data.ndim != 3:
        raise ValueError(f"Error: expected a 3D NIfTI file. Got shape '{nii_data.shape}' instead.")

    # Define valid swaps and corresponding axis reordering
    valid_swaps = {
        ("axial", "coronal"): (0, 2, 1),  # swap Y and Z → (X, Z, Y)
        ("coronal", "axial"): (0, 2, 1),  # swap back (X, Z, Y) → (X, Y, Z)
        ("axial", "sagittal"): (2, 1, 0), # swap X and Z → (Z, Y, X)
        ("sagittal", "axial"): (2, 1, 0), # swap back (Z, Y, X) → (X, Y, Z)
        ("coronal", "sagittal"): (1, 0, 2), # swap X and Y → (Y, X, Z)
        ("sagittal", "coronal"): (1, 0, 2)  # swap back (Y, X, Z) → (X, Y, Z)
    }

    # Define necessary rotations (90 degrees clockwise)
    rotation_axes = {
        ("axial", "coronal"): (1, 2),  # Rotate around X
        ("coronal", "axial"): (1, 2),  # Rotate back around X
        ("axial", "sagittal"): (0, 2), # Rotate around Y
        ("sagittal", "axial"): (0, 2), # Rotate back around Y
        ("coronal", "sagittal"): (0, 1), # Rotate around Z
        ("sagittal", "coronal"): (0, 1)  # Rotate back around Z
    }

    # check if the provided view swap is valid
    if (source_view, target_view) not in valid_swaps:
        raise ValueError("Error: Invalid view swap. Choices are: axial ↔ coronal, axial ↔ sagittal, coronal ↔ sagittal.")

    # get the new axis order
    new_axes = valid_swaps[(source_view, target_view)]

    # swap axes in data
    swapped_data = np.transpose(nii_data, new_axes)

    # Apply the required 90-degree rotation
    rot_axes = rotation_axes[(source_view, target_view)]
    swapped_data = np.rot90(swapped_data, k=1, axes=rot_axes)  # k=1 means 90 degrees clockwise

    # Adjust affine matrix to match the new view
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3][new_axes, :]  # Adjust rotation
    new_affine[:3, 3] = affine[:3, 3]  # Adjust translation
    new_affine[3, 3] = 1  # Ensure homogeneous coordinates remain intact

    # update header with new shape
    new_header = header.copy()
    new_header.set_data_shape(swapped_data.shape)

    # create new NIfTI image with swapped data
    swapped_img = nib.Nifti1Image(swapped_data, new_affine, header=new_header)

    # extract filename prefix
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")

    # save the swapped image
    swapped_filename = os.path.join(output_path, f"{prefix}_swapped_{source_view}_to_{target_view}.nii.gz")
    nib.save(swapped_img, swapped_filename)

    # debug print
    if debug:
        print(f"\nInput file: '{nii_path}'\nOutput path: '{output_path}'")
        print(f"Original shape: {nii_data.shape} | Swapped shape: {swapped_data.shape}")
        print(f"View swapped from {source_view} to {target_view}")
        print(f"Swapped NIfTI saved at: {swapped_filename}")


def mip(nii_path: str,
        output_path: str,
        window_size: int = 10,
        view: str = "axial",
        debug: bool = False) -> None:
    """
    Generates a 3D Maximum Intensity Projection (MIP) from a NIfTI file. Save the output file as:

        <Nifti FILENAME>_mip_<VIEW>.nii.gz

    :param nii_path: path to the input .nii.gz file with shape (X, Y, Z).
    :param output_path: path where the MIP .nii.gz file will be saved.
    :param window_size: number of slices to merge for the MIP.
    :param view: "axial" (default) -> performs MIP along the Z-axis.
                 "coronal" -> performs MIP along the Y-axis.
                 "sagittal" -> performs MIP along the X-axis.

    :raises FileNotFoundError: if the input NIfTI file does not exist.
    :raises ValueError: if the file is empty, has invalid dimensions, or if the axis is incorrect.

    Example:

        from nidataset.Volume import mip

        # define paths
        nii_path = "path/to/input_image.nii.gz"
        output_path = "path/to/output_directory"

        # choose the anatomical view ('axial', 'coronal', or 'sagittal')
        view = "axial"

        # run the function
        mip(nii_path=nii_path,
            output_path=output_path,
            window_size=20,
            view=view,
            debug=True)
    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file, but got '{nii_path}'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the NIfTI file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    affine = nii_img.affine  # preserve transformation matrix

    # validate NIfTI data dimensions
    if nii_data.ndim != 3:
        raise ValueError(f"Error: expected a 3D NIfTI file. Got shape '{nii_data.shape}' instead.")

    # define projection axis
    view_mapping = {"axial": 2, "coronal": 1, "sagittal": 0}
    if view not in view_mapping:
        raise ValueError("Error: axis must be 'axial', 'coronal', or 'sagittal'.")
    axis_index = view_mapping[view]

    # define prefix as the nii.gz filename
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")

    # initialize MIP output volume
    mip_data = np.zeros_like(nii_data)

    # iterate over each slice along the chosen axis
    tqdm_desc = f"Processing MIP ({view}, {window_size} slices) for {prefix}"
    for i in tqdm(range(nii_data.shape[axis_index]), desc=tqdm_desc, unit="slice"):
        # define the range of slices from i - window_size to i + window_size
        start_slice = max(0, i - window_size)  # ensure range doesn't go below 0
        end_slice = min(nii_data.shape[axis_index], i + window_size + 1)  # ensure range doesn't exceed data

        # extract the subvolume for projection
        if view == "axial":
            subvolume = nii_data[:, :, start_slice:end_slice]
            mip_result = np.max(subvolume, axis=2)
            mip_data[:, :, i] = mip_result
        elif view == "coronal":
            subvolume = nii_data[:, start_slice:end_slice, :]
            mip_result = np.max(subvolume, axis=1)
            mip_data[:, i, :] = mip_result
        elif view == "sagittal":
            subvolume = nii_data[start_slice:end_slice, :, :]
            mip_result = np.max(subvolume, axis=0)
            mip_data[i, :, :] = mip_result

    # create a new NIfTI image with the projected data
    mip_image = nib.Nifti1Image(mip_data, affine)

    # save the new image to a file
    mip_filename = os.path.join(output_path, f"{prefix}_mip_{view}.nii.gz")
    nib.save(mip_image, mip_filename)

    if debug:
        print(f"\nMIP saved at: {mip_filename}")


def mip_from_dataset(nii_folder: str, 
                     output_path: str, 
                     window_size: int = 10, 
                     view: str = "axial",
                     saving_mode: str = "case", 
                     debug: bool = False) -> None:
    """
    Generates 3D Maximum Intensity Projections (MIP) from all NIfTI files in a dataset folder. Save the output file as:

        <Nifti FILENAME>_mip_<VIEW>.nii.gz

    :param nii_folder: path to the folder containing .nii.gz files.
    :param output_path: path where the MIP .nii.gz files will be saved.
    :param window_size: number of slices to merge for the MIP.
    :param view: "axial" (default) -> performs MIP along the Z-axis.
                 "coronal" -> performs MIP along the Y-axis.
                 "sagittal" -> performs MIP along the X-axis.
    :param saving_mode: "case" -> creates a folder for each case.
                        "view" -> saves all MIP files inside a single folder.
    :param debug: if True, prints additional information about the process.

    :raises FileNotFoundError: if the dataset folder does not exist or contains no .nii.gz files.
    :raises ValueError: if an invalid view or saving_mode is provided.

    Example:

        from nidataset.Volume import mip_from_dataset

        # define paths
        nii_folder = "path/to/dataset"
        output_path = "path/to/output_directory"

        # choose the anatomical view ('axial', 'coronal', or 'sagittal')
        view = "axial"

        # run the function
        mip_from_dataset(nii_folder=nii_folder, 
                 output_path=output_path, 
                 window_size=20, 
                 view=view, 
                 saving_mode="case", 
                 debug=True)

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
    if view not in ["axial", "coronal", "sagittal"]:
        raise ValueError("Error: view must be 'axial', 'coronal', or 'sagittal'.")
    if saving_mode not in ["case", "view"]:
        raise ValueError("Error: saving_mode must be either 'case' or 'view'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create a single folder for the chosen view if using "view" mode
    if saving_mode == "view":
        view_output_dir = os.path.join(output_path, view)
        os.makedirs(view_output_dir, exist_ok=True)

    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Processing NIfTI files", unit="file"):
        # nii.gz file path
        nii_path = os.path.join(nii_folder, nii_file)

        # extract the filename prefix (case ID)
        prefix = os.path.basename(nii_path).replace(".nii.gz", "")

        # update tqdm description with the current file prefix
        tqdm.write(f"Processing: {prefix}")

        # determine the appropriate output folder
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix, view)
            os.makedirs(case_output_dir, exist_ok=True)
            mip(nii_path, case_output_dir, window_size, view, debug=False)
        else:
            mip(nii_path, view_output_dir, window_size, view, debug=False)

    if debug:
        print(f"\nMIP processing completed for all files in '{nii_folder}'")


def extract_bounding_boxes(mask_path: str, 
                           output_path: str, 
                           voxel_size: tuple = (3.0, 3.0, 3.0), 
                           volume_threshold: float = 1000.0, 
                           mask_value: int = 1, 
                           debug: bool = False) -> None:
    """
    Extracts 3D bounding boxes from a segmentation mask and saves the bounding box annotation as a NIfTI file:

        <Nifti FILENAME>_bounding_boxes.nii.gz

    :param mask_path: path to the input 3D segmentation mask (.nii.gz).
    :param output_path: path where the bounding box annotation will be saved.
    :param voxel_size: size of a voxel in mm (x, y, z).
    :param volume_threshold: minimum volume (mm³) for connected components to be considered.
    :param mask_value: value in the mask representing the target region.
    :param debug: if True, prints additional information about bounding box extraction.

    :raises FileNotFoundError: if the mask file does not exist.
    :raises ValueError: if the input file is not a valid 3D NIfTI image.

    Example:

        from nidataset.Volume import extract_bounding_boxes

        # define paths
        mask_path = "path/to/input_mask.nii.gz"
        output_path = "path/to/output_directory"

        # run the function
        extract_bounding_boxes(mask_path=mask_path, 
                               output_path=output_path)

    """

    # check if the input mask file exists
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Error: the input file '{mask_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not mask_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file, but got '{mask_path}'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the NIfTI mask file
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # extract filename prefix
    prefix = os.path.basename(mask_path).replace(".nii.gz", "")

    # filter the mask to retain only the desired label
    binary_mask = (mask_data == mask_value).astype(np.uint8)

    # find connected components in the binary mask
    labeled_components, num_components = ndi.label(binary_mask)

    # initialize bounding box storage
    bounding_boxes = []

    # iterate over detected components
    for label in tqdm(range(1, num_components + 1), desc=f"Processing {prefix} (Bounding Boxes)", unit="box"):
        # compute volume of the connected component
        component_volume_mm3 = np.sum(labeled_components == label) * np.prod(voxel_size)

        # apply volume threshold filter
        if component_volume_mm3 >= volume_threshold:
            # get min/max coordinates of the bounding box
            component_indices = np.argwhere(labeled_components == label)
            min_coords = component_indices.min(axis=0)
            max_coords = component_indices.max(axis=0)

            # store bounding box
            bounding_boxes.append((tuple(min_coords), tuple(max_coords)))

    # create an empty image for the bounding box annotation
    bounding_box_image = np.zeros_like(mask_data, dtype=np.uint8)

    # draw bounding boxes on the new image
    for bbox in bounding_boxes:
        min_x, min_y, min_z = bbox[0]
        max_x, max_y, max_z = bbox[1]
        bounding_box_image[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1] = 255

    # save the bounding box NIfTI file
    bbox_nifti = nib.Nifti1Image(bounding_box_image, mask_img.affine)
    bbox_filename = os.path.join(output_path, f"{prefix}_bounding_boxes.nii.gz")
    nib.save(bbox_nifti, bbox_filename)

    # debug print
    if debug:
        print(f"\nInput file: '{mask_path}'\nOutput path: '{bbox_filename}'\nTotal bounding boxes extracted: {len(bounding_boxes)}")


def extract_bounding_boxes_from_dataset(mask_folder: str, 
                                        output_path: str, 
                                        voxel_size: tuple = (3.0, 3.0, 3.0), 
                                        volume_threshold: float = 1000.0, 
                                        mask_value: int = 1,
                                        save_stats: bool = True,
                                        debug: bool = False) -> None:
    """
    Extracts 3D bounding boxes from all segmentation masks in a dataset folder and saves them as NIfTI files:

        <Nifti FILENAME>_bounding_boxes.nii.gz

    Also generates a statistics CSV file containing the number of bounding boxes per file.

    :param mask_folder: path to the folder containing .nii.gz segmentation masks.
    :param output_path: path where the bounding box annotations will be saved.
    :param voxel_size: size of a voxel in mm (x, y, z).
    :param volume_threshold: minimum volume (mm³) for connected components to be considered.
    :param mask_value: value in the mask representing the target region.
    :param save_stats: if True, saves a CSV file with FILENAME and NUM_BOUNDING_BOXES information per case.
    :param debug: if True, prints additional information about bounding box extraction.

    :raises FileNotFoundError: if the dataset folder does not exist or contains no .nii.gz files.

    Example:

        from nidataset.Volume import extract_bounding_boxes_from_dataset

        # define paths
        mask_folder = "path/to/masks"
        output_path = "path/to/output_directory"

        # run the function
        extract_bounding_boxes(mask_folder=mask_folder, 
                               output_path=output_path)

    """

    # check if the dataset folder exists
    if not os.path.isdir(mask_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{mask_folder}' does not exist.")

    # get all .nii.gz mask files in the dataset folder
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not mask_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{mask_folder}'.")

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize statistics tracking
    stats = []
    total_bounding_boxes = 0
    stats_file = os.path.join(output_path, "bounding_boxes_stats.csv") if save_stats else None

    # iterate over mask files with tqdm progress bar
    for mask_file in tqdm(mask_files, desc="Processing NIfTI mask files", unit="file"):
        # mask file path
        mask_path = os.path.join(mask_folder, mask_file)

        # extract bounding box count for statistics
        try:
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            binary_mask = (mask_data == mask_value).astype(np.uint8)
            labeled_components, num_components = ndi_label(binary_mask)

            # count valid bounding boxes based on volume threshold
            num_bboxes = 0
            for label in range(1, num_components + 1):
                component_volume_mm3 = np.sum(labeled_components == label) * np.prod(voxel_size)
                if component_volume_mm3 >= volume_threshold:
                    num_bboxes += 1

            total_bounding_boxes += num_bboxes
            if save_stats:
                stats.append([mask_file, num_bboxes])

        except Exception as e:
            tqdm.write(f"Error processing {mask_file} for statistical analysis: {e}")
            continue  # skip this file if an error occurs

        # call extract_bounding_boxes function (without modifications)
        extract_bounding_boxes(mask_path, output_path, voxel_size, volume_threshold, mask_value, debug=False)

    # save statistics if enabled
    if save_stats:
        with open(stats_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["FILENAME", "NUM_BOUNDING_BOXES"])
            writer.writerows(stats)
            writer.writerow(["TOTAL_BOUNDING_BOXES", total_bounding_boxes])
        
        print(f"\nBounding box statistics saved in: '{stats_file}'")
    
    # debug print
    if debug:
        print(f"\nInput folder: '{mask_folder}'\nOutput path: '{output_path}'")
        print(f"Total files processed: {len(mask_files)} | Total bounding boxes extracted: {total_bounding_boxes}")


def generate_brain_mask(nii_path: str, 
                        output_path: str, 
                        threshold: tuple = None,
                        closing_radius: int = 3,
                        debug: bool = False) -> None:
    """
    Generates a brain mask from a brain CTA scan in NIfTI format saved as:

        <Nifti FILENAME>_brain_mask.nii.gz

    :param nii_path: path to the input brain CTA .nii.gz file.
    :param output_path: path where the brain mask NIfTI file will be saved.
    :param threshold: tuple (low, high) to segment the brain based on intensity values.
                      if None, Otsu's thresholding is applied automatically.
    :param closing_radius: radius for morphological closing to fill small gaps.
    :param debug: if True, prints additional information about the mask generation.

    :raises FileNotFoundError: if the input NIfTI file does not exist.
    :raises ValueError: if the file is empty or has invalid dimensions.

    Example:

        from nidataset.Volume import generate_brain_mask

        # define paths
        nii_path = "path/to/input_image.nii.gz"
        output_path = "path/to/output_directory"

        # run the function
        extract_bounding_boxes(mask_folder=mask_folder, 
                               output_path=output_path)


    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file, but got '{nii_path}'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the CTA NIfTI file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    affine = nii_img.affine  # preserve transformation matrix

    # validate NIfTI data dimensions
    if nii_data.ndim != 3:
        raise ValueError(f"Error: expected a 3D NIfTI file. Got shape '{nii_data.shape}' instead.")

    # validate threshold dimensions
    if len(threshold) != 2:
        raise ValueError(f"Error: expected two threshold values, but got shape {len(threshold)}")

    # automatically determine threshold using Otsu if none provided
    if threshold is None:
        otsu_thresh = threshold_otsu(nii_data[nii_data > 0])  # ignore background
        lower_bound = otsu_thresh * 0.5  # relax lower bound to capture soft tissue
        upper_bound = otsu_thresh * 2.0  # allow brighter regions
        if debug:
            print(f"\nUsing Otsu's threshold: {otsu_thresh:.2f}\nAdjusted range: ({lower_bound:.2f}, {upper_bound:.2f})")
    else:
        lower_bound, upper_bound = threshold

    # apply thresholding to create an initial brain mask
    binary_mask = (nii_data >= lower_bound) & (nii_data <= upper_bound)

    # fill small holes inside the mask
    binary_mask_filled = ndi.binary_fill_holes(binary_mask)

    # apply morphological closing to refine the mask
    brain_mask = binary_closing(binary_mask_filled, ball(closing_radius))

    # extract the largest connected component (assumed to be the brain)
    labeled_mask, num_components = ndi.label(brain_mask)
    component_sizes = np.bincount(labeled_mask.ravel())
    largest_component_label = np.argmax(component_sizes[1:]) + 1  # ignore background label (0)
    brain_mask = labeled_mask == largest_component_label

    # convert mask to uint8 format (0 and 1)
    brain_mask = brain_mask.astype(np.uint8)

    # create new NIfTI image with the brain mask
    brain_mask_nifti = nib.Nifti1Image(brain_mask, affine)

    # extract filename prefix
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")

    # save the new brain mask image
    mask_filename = os.path.join(output_path, f"{prefix}_brain_mask.nii.gz")
    nib.save(brain_mask_nifti, mask_filename)

    # debug print
    if debug:
        print(f"\nInput file: '{nii_path}'\nOutput path: '{output_path}'\nBrain mask saved at: {mask_filename}")


def generate_brain_mask_from_dataset(nii_folder: str, 
                                     output_path: str, 
                                     threshold: tuple = None,
                                     closing_radius: int = 3,
                                     debug: bool = False) -> None:
    """
    Generates brain masks for all brain CTA scans in a dataset folder and saves them as NIfTI files with name:

        <Nifti FILENAME>_brain_mask.nii.gz

    :param nii_folder: path to the folder containing .nii.gz brain CTA scans.
    :param output_path: path where the brain mask NIfTI files will be saved.
    :param threshold: tuple (low, high) to segment the brain based on intensity values.
                      if None, Otsu's thresholding is applied automatically.
    :param closing_radius: radius for morphological closing to fill small gaps.
    :param debug: if True, prints additional information about the mask generation.

    :raises FileNotFoundError: if the dataset folder does not exist or contains no .nii.gz files.

    Example:

        from nidataset.Volume import generate_brain_mask_from_dataset

        # define paths
        nii_folder = "path/to/dataset"
        output_path = "path/to/output_directory"

        # run the function
        generate_brain_mask_from_dataset(nii_folder=nii_folder, 
                                         output_path=output_path)

    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{nii_folder}' does not exist.")

    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{nii_folder}'.")

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # iterate over NIfTI files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Processing Brain CTA files", unit="file"):
        # construct full file path
        nii_path = os.path.join(nii_folder, nii_file)

        # call generate_brain_mask function
        generate_brain_mask(nii_path, output_path, threshold, closing_radius, debug=debug)

    # debug print
    if debug:
        print(f"\nInput folder: '{nii_folder}'\nOutput path: '{output_path}'")
        print(f"Total brain masks generated: {len(nii_files)}")


def crop_and_pad(nii_path: str,
                 output_path: str,
                 target_shape: tuple = (128, 128, 128),
                 debug: bool = False) -> None:
    """
    Finds the minimum bounding box around a CTA scan, resizes it to a target shape, and preserves spatial orientation. Save the file with name:

        <Nifti FILENAME>_cropped_padded.nii.gz

    :param nii_path: path to the input CTA .nii.gz file.
    :param output_path: path where the cropped and padded CTA file will be saved.
    :param target_shape: desired (X, Y, Z) shape after padding or cropping.
    :param debug: if True, prints additional information about the processing.

    :raises FileNotFoundError: if the input file does not exist.
    :raises ValueError: if the file is empty or has invalid dimensions.

    Example:

        from nidataset.Volume import crop_and_pad

        # define paths
        nii_path = "path/to/input_image.nii.gz"
        output_path = "path/to/output_directory"

        # run the function
        crop_and_pad(nii_path=nii_path, 
                     output_path=output_path)

    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: The input file '{nii_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: Invalid file format. Expected a '.nii.gz' file, but got '{nii_path}'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the CTA NIfTI file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    affine = nii_img.affine.copy()  # preserve original transformation matrix
    header = nii_img.header.copy()  # preserve original header

    # validate NIfTI data dimensions
    if nii_data.ndim != 3:
        raise ValueError(f"Error: Expected a 3D NIfTI file. Got shape '{nii_data.shape}' instead.")

    # validate target shape dimensions
    if len(target_shape) != 3:
        raise ValueError(f"Error: Expected a 3D target shape, but got shape {len(target_shape)}")

    # find the bounding box of nonzero voxels
    coords = np.argwhere(nii_data > 0)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)

    # crop the image to the bounding box
    cropped_data = nii_data[min_coords[0]:max_coords[0]+1, 
                            min_coords[1]:max_coords[1]+1, 
                            min_coords[2]:max_coords[2]+1]

    # get the new cropped shape
    cropped_shape = cropped_data.shape

    # calculate the padding required to reach the target shape
    pad_x = max(0, target_shape[0] - cropped_shape[0])
    pad_y = max(0, target_shape[1] - cropped_shape[1])
    pad_z = max(0, target_shape[2] - cropped_shape[2])

    # calculate how much to crop if the volume is too large
    crop_x = max(0, cropped_shape[0] - target_shape[0])
    crop_y = max(0, cropped_shape[1] - target_shape[1])
    crop_z = max(0, cropped_shape[2] - target_shape[2])

    # apply cropping if needed
    if crop_x > 0:
        cropped_data = cropped_data[crop_x//2:-(crop_x//2 or None), :, :]
    if crop_y > 0:
        cropped_data = cropped_data[:, crop_y//2:-(crop_y//2 or None), :]
    if crop_z > 0:
        cropped_data = cropped_data[:, :, crop_z//2:-(crop_z//2 or None)]

    # apply padding if needed
    pad_width = ((pad_x//2, pad_x - pad_x//2),
                 (pad_y//2, pad_y - pad_y//2),
                 (pad_z//2, pad_z - pad_z//2))
    padded_data = np.pad(cropped_data, pad_width, mode='constant', constant_values=0)

    # adjust the affine matrix to maintain spatial alignment
    # offset is calculated to match the original position of the cropped region
    pad_offset = np.array([pad_width[0][0], pad_width[1][0], pad_width[2][0]])
    crop_offset = np.array([min_coords[0], min_coords[1], min_coords[2]])

    # adjust affine transformation: shift back by cropping, then by padding
    new_affine = affine.copy()
    new_affine[:3, 3] += np.dot(new_affine[:3, :3], crop_offset - pad_offset)

    # adjust header
    new_header_image = header.copy()
    new_header_image.set_data_shape(padded_data.shape)

    # create new NIfTI image with the transformed data
    processed_image = nib.Nifti1Image(padded_data, new_affine, header=new_header_image)
    processed_image.set_qform(new_affine)
    processed_image.set_sform(new_affine)

    # extract filename prefix
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")

    # save the processed image
    processed_filename = os.path.join(output_path, f"{prefix}_cropped_padded.nii.gz")
    nib.save(processed_image, processed_filename)

    # debug print
    if debug:
        print(f"\nInput File: '{nii_path}'\nOutput Path: '{output_path}'")
        print(f"Original Shape: {nii_data.shape} | Cropped Shape: {cropped_shape} | Final Shape: {padded_data.shape}")
        print(f"Processed CTA Saved at: {processed_filename}")


def crop_and_pad_from_dataset(nii_folder: str, 
                              output_path: str, 
                              target_shape: tuple = (128, 128, 128), 
                              save_stats: bool = False) -> None:
    """
    Processes all CTA scans in a dataset folder, applies crop_and_pad, and saves results with name:

        <Nifti FILENAME>_cropped_padded.nii.gz

    :param nii_folder: Path to the folder containing .nii.gz CTA files.
    :param output_path: Path where the cropped and padded CTA files will be saved.
    :param target_shape: Desired (X, Y, Z) shape after padding or cropping.
    :param save_stats: If True, saves a CSV file with FILENAME, ORIGINAL_SHAPE, FINAL_SHAPE.

    :raises FileNotFoundError: If the dataset folder does not exist or contains no .nii.gz files.

    Example:

        from nidataset.Volume import crop_and_pad_from_dataset

        # define paths
        nii_folder = "path/to/dataset"
        output_path = "path/to/output_directory"

        # run the function
        crop_and_pad_from_dataset(nii_folder=nii_folder, 
                                  output_path=output_path)
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: The dataset folder '{nii_folder}' does not exist.")

    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: No .nii.gz files found in '{nii_folder}'.")

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize statistics tracking
    stats = []
    stats_file = os.path.join(output_path, "crop_pad_stats.csv") if save_stats else None

    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Processing NIfTI files", unit="file"):
        # nii.gz file path
        nii_path = os.path.join(nii_folder, nii_file)

        # extract the filename prefix (case ID)
        prefix = os.path.basename(nii_path).replace(".nii.gz", "")

        # update tqdm description with the current file prefix
        tqdm.write(f"Processing: {prefix}")

        # get the original shape for statistics
        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
            original_shape = nii_data.shape
        except Exception as e:
            tqdm.write(f"Error processing {nii_file}: {e}")
            continue  # skip this file if an error occurs

        # call crop_and_pad function
        crop_and_pad(nii_path, output_path, target_shape, debug=False)

        # get the final shape for statistics
        processed_filename = os.path.join(output_path, f"{prefix}_cropped_padded.nii.gz")
        try:
            processed_img = nib.load(processed_filename)
            processed_data = processed_img.get_fdata()
            final_shape = processed_data.shape
        except Exception as e:
            tqdm.write(f"Error loading processed file {processed_filename}: {e}")
            continue  # skip this file if an error occurs

        # store statistics
        if save_stats:
            stats.append([nii_file, original_shape, final_shape])

    # save statistics if enabled
    if save_stats:
        with open(stats_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["FILENAME", "ORIGINAL_SHAPE", "FINAL_SHAPE"])
            writer.writerows(stats)
        
        print(f"\nCrop and pad statistics saved in: '{stats_file}'")


def skull_CTA(nii_path: str,
              output_path: str,
              f_value: float = 0.1,
              clip_value: tuple = (0, 200),
              cleanup: bool = False,
              debug: bool = False) -> None:
    """
    Performs a skull-stripping pipeline designed for CTA (thresholding, smoothing, FSL BET, and clipping) on a single NIfTI file.

    :implementation note:
        CTAs has to be already centered to the brain area (no robust_fov is applied in this pipeline to ensure the input dimension is kept).
        FSL command line needs to be installed via official site (https://fsl.fmrib.ox.ac.uk/fsldownloads_registration/).
        The file that will use this function needs to be launched from terminal: `python3 main.py`, where `main.py` use this function.

    :param nii_path: Path to the input .nii.gz file with shape (X, Y, Z).
    :param output_path: Directory where intermediate and final outputs will be stored.
    :param f_value: Fractional intensity threshold for BET (skull-stripping).
    :param clip_value:   Minimum and maximum intensity values for clipping, e.g. (0, 200).
    :param cleanup: If True, removes intermediate files (thresholded and skulled images). Mask and clipped brain outputs are always kept.
    :param debug: If True, prints additional debugging information.

    :raises FileNotFoundError: If nii_path does not exist.
    :raises ValueError: If the file is not a .nii.gz, or if data is not 3D.
    """

    # validate input path
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure data type
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file, but got '{nii_path}'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # cuild intermediate paths
    base_name    = os.path.basename(nii_path).replace(".nii.gz", "")
    th_img       = os.path.join(output_path, f"{base_name}_th.nii.gz")
    th_sm_img    = os.path.join(output_path, f"{base_name}_th_sm.nii.gz")
    th_sm_th_img = os.path.join(output_path, f"{base_name}_th_sm_th.nii.gz")
    skulled_img  = os.path.join(output_path, f"{base_name}.skulled.nii.gz")
    mask_img     = os.path.join(output_path, f"{base_name}.skulled_mask.nii.gz")
    clipped_img  = os.path.join(output_path, f"{base_name}.skulled.clipped.nii.gz")

    # threshold [0-100], smoothing, threshold [0-100]
    try:
        subprocess.run(["fslmaths", nii_path, "-thr", "0", "-uthr", "100", th_img], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["fslmaths", th_img, "-s", "1", th_sm_img], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["fslmaths", th_sm_img, "-thr", "0", "-uthr", "100", th_sm_th_img], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # BET skull stripping (makes the skulled image + mask)
        subprocess.run([
            "bet", th_sm_th_img, skulled_img, "-R",
            "-f", str(f_value), "-g", "0", "-m"
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FSL command failed for '{nii_path}' with error: {e.stderr.decode()}")

    # load skulled image, clip intensities to desired values, save final .nii.gz
    nii_skulled = nib.load(skulled_img)
    skulled_data = nii_skulled.get_fdata()
    clipped_data = np.clip(skulled_data, clip_value[0], clip_value[0])  # clip to desired values
    clipped_nii  = nib.Nifti1Image(clipped_data, nii_skulled.affine, nii_skulled.header)
    nib.save(clipped_nii, clipped_img)

    # optional cleanup
    if cleanup:
        # remove intermediate files except mask and clipped images
        for tmp_file in [th_img, th_sm_img, th_sm_th_img, skulled_img]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        if debug:
            print("Intermediate files have been removed.")

    if debug:
        print(f"\nSkull-stripped image saved at: {clipped_img}\n"
            f"Skull mask saved at: {mask_img}")


def skull_CTA_from_dataset(nii_folder: str,
                           output_path: str,
                           f_value: float = 0.1,
                           clip_value: tuple = (0, 200),
                           cleanup: bool = False,
                           saving_mode: str = "case",
                           debug: bool = False) -> None:
    """
    Performs a skull-stripping pipeline designed for CTA (thresholding, smoothing, FSL BET, and clipping) on the NIfTI files inside the input folder.

    :implementation note:
        CTAs has to be already centered to the brain area (no robust_fov is applied in this pipeline to ensure the input dimension is kept).
        FSL command line needs to be installed via official site (https://fsl.fmrib.ox.ac.uk/fsldownloads_registration/).
        The file that will use this function needs to be launched from terminal: `python3 main.py`, where `main.py` use this function.

    :param nii_folder:  Directory containing .nii.gz files to be processed.
    :param output_path: Directory where the skull-stripped .nii.gz files will be saved.
    :param f_value:     Fractional intensity threshold for BET (skull-stripping).
    :param clip_value:  Minimum and maximum intensity values for clipping, e.g. (0, 200).
    :param cleanup:     If True, removes intermediate files (thresholded and skulled images). Mask and clipped images are always retained.
    :param saving_mode: "case" -> creates a dedicated subfolder for each .nii.gz file.
                        "view" -> saves all the results in a single subfolder called 'skulled'.
    :param debug:       If True, prints additional information about the process.

    :raises FileNotFoundError: If the dataset folder does not exist or contains no .nii.gz files.
    :raises ValueError:        If an invalid saving_mode is provided.
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{nii_folder}' does not exist.")

    # retrieve all .nii.gz files
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]
    if not nii_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{nii_folder}'.")

    # validate saving_mode
    if saving_mode not in ["case", "view"]:
        raise ValueError("Error: saving_mode must be either 'case' or 'view'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # for "view" mode, create a single folder to store all outputs
    if saving_mode == "view":
        view_output_dir = os.path.join(output_path, "skulled")
        os.makedirs(view_output_dir, exist_ok=True)
    else:
        view_output_dir = None

    # process files with a progress bar
    for nii_file in tqdm(nii_files, desc="Skull-stripping NIfTI files", unit="file"):
        nii_path = os.path.join(nii_folder, nii_file)
        prefix   = os.path.splitext(os.path.splitext(nii_file)[0])[0]  # remove .nii.gz

        if debug:
            print(f"Processing: {prefix}")

        # if saving_mode = "case", create one subfolder for each file
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix)
            os.makedirs(case_output_dir, exist_ok=True)
            skull_CTA(
                nii_path=nii_path,
                output_path=case_output_dir,
                f_value=f_value,
                clip_value=clip_value,
                cleanup=cleanup,
                debug=debug
            )

        else:  # saving_mode = "view"
            skull_CTA(
                nii_path=nii_path,
                output_path=view_output_dir,
                f_value=f_value,
                clip_value=clip_value,
                cleanup=cleanup,
                debug=debug
            )

    if debug:
        print(f"Skull-stripping completed for all files in '{nii_folder}'.")

    
