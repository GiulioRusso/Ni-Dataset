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
    Swaps the anatomical view of a 3D NIfTI image by reordering axes, applying 
    a 90-degree rotation, and updating the affine matrix to preserve orientation.
    Saves:

        <PREFIX>_swapped_<SOURCE VIEW>_to_<TARGET VIEW>.nii.gz

    .. note::
       - Supports swaps between axial, coronal, and sagittal views.
       - Each swap applies the necessary axis permutation and 90-degree rotation.
       - The affine matrix is updated to reflect the new orientation.
       - Input files must be 3D NIfTI images.
       - Output directory is created if it does not exist.

    :param nii_path:
        Path to the input 3D NIfTI (.nii.gz) file.

    :param output_path:
        Directory where the swapped NIfTI file will be saved. Created if missing.

    :param source_view:
        Current anatomical view of the volume. Must be one of ``"axial"``, 
        ``"coronal"``, or ``"sagittal"``.

    :param target_view:
        Desired anatomical view. Must be one of ``"axial"``, ``"coronal"``, 
        or ``"sagittal"``.

    :param debug:
        If ``True``, prints detailed information about the input shape, swapped 
        shape, and saved file path.

    :raises FileNotFoundError:
        If the input NIfTI file does not exist.

    :raises ValueError:
        If the input file is not 3D, has invalid dimensions, or if the views 
        are not among the allowed choices.

    Example
    -------
    >>> from nidataset.volume import swap_nifti_views
    >>>
    >>> swap_nifti_views(
    ...     nii_path="dataset/CTA_raw/case_01.nii.gz",
    ...     output_path="output/CTA_swapped/",
    ...     source_view="axial",
    ...     target_view="sagittal",
    ...     debug=True
    ... )
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


def extract_bounding_boxes(mask_path: str, 
                           output_path: str, 
                           voxel_size: tuple = (3.0, 3.0, 3.0), 
                           volume_threshold: float = 1000.0, 
                           mask_value: int = 1, 
                           debug: bool = False) -> None:
    """
    Extracts 3D bounding boxes from a segmentation mask by identifying connected
    components, filtering out small regions, and generating a bounding-box 
    annotation volume. Saves:

        <PREFIX>_bounding_boxes.nii.gz

    .. note::
       - Connected components are detected from the input mask using the value 
         specified in ``mask_value``.
       - Components with physical volume below ``volume_threshold`` (mm³) 
         are discarded.
       - The output file contains a binary volume where each bounding box is 
         filled with the value ``255``.
       - The output directory is created automatically if missing.

    :param mask_path:
        Path to the input 3D segmentation mask (``.nii.gz``). The mask must be 
        a single-label or multi-label volume containing the target region.

    :param output_path:
        Directory where the bounding box annotation NIfTI file will be saved.

    :param voxel_size:
        Physical voxel dimensions in millimeters, given as ``(sx, sy, sz)``. 
        Used to compute component volumes.

    :param volume_threshold:
        Minimum physical volume (in mm³) required for a connected component
        to be considered valid.

    :param mask_value:
        Integer label in the mask corresponding to the region of interest.

    :param debug:
        If ``True``, prints detailed information about detected components,
        bounding box sizes, and final output path.

    :raises FileNotFoundError:
        If the mask file does not exist.

    :raises ValueError:
        If the file is not a valid ``.nii.gz`` or is not a 3D NIfTI image.

    Example
    -------
    >>> from nidataset.volume import extract_bounding_boxes
    >>>
    >>> extract_bounding_boxes(
    ...     mask_path="dataset/masks/case_01_mask.nii.gz",
    ...     output_path="output/bounding_boxes/",
    ...     voxel_size=(3.0, 3.0, 3.0),
    ...     volume_threshold=1500.0,
    ...     mask_value=1,
    ...     debug=True
    ... )
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


def extract_bounding_boxes_dataset(mask_folder: str, 
                                   output_path: str, 
                                   voxel_size: tuple = (3.0, 3.0, 3.0), 
                                   volume_threshold: float = 1000.0, 
                                   mask_value: int = 1,
                                   save_stats: bool = True,
                                   debug: bool = False) -> None:
    """
    Extracts 3D bounding boxes from all segmentation masks in a dataset folder and
    saves a NIfTI file for each case:

        <PREFIX>_bounding_boxes.nii.gz

    Optionally generates a CSV file reporting the number of bounding boxes per mask.

    .. note::
       - Only connected components larger than ``volume_threshold`` (mm³) are kept.
       - Bounding boxes are extracted from voxels equal to ``mask_value``.
       - Output directory is created if it does not exist.
       - If ``save_stats`` is True, a CSV file named ``bounding_boxes_stats.csv`` is saved.
       - Input masks must be 3D NIfTI images (``.nii.gz``).

    :param mask_folder:
        Path to the directory containing 3D segmentation masks (``.nii.gz`` files).

    :param output_path:
        Directory where bounding box NIfTI files (and statistics CSV, if enabled)
        will be saved.

    :param voxel_size:
        Physical voxel size in millimeters as ``(x, y, z)`` used to compute
        connected component volumes.

    :param volume_threshold:
        Minimum component volume (in mm³) required for a bounding box to be included.

    :param mask_value:
        Integer value inside the mask that represents the target region.

    :param save_stats:
        If ``True``, saves a CSV summarizing the number of bounding boxes per file.

    :param debug:
        If ``True``, prints internal diagnostic information and summaries.

    :raises FileNotFoundError:
        If ``mask_folder`` does not exist or contains no ``.nii.gz`` files.

    Example
    -------
    >>> from nidataset.volume import extract_bounding_boxes_dataset
    >>>
    >>> extract_bounding_boxes_dataset(
    ...     mask_folder="dataset/CTA_masks/",
    ...     output_path="output/CTA_bounding_boxes/",
    ...     voxel_size=(3.0, 3.0, 3.0),
    ...     volume_threshold=1500.0,
    ...     mask_value=1,
    ...     save_stats=True,
    ...     debug=True
    ... )
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
    Generates a brain mask from a brain CTA scan in NIfTI format and saves the file as:

        <PREFIX>_brain_mask.nii.gz

    .. note::
       - If ``threshold`` is None, Otsu’s thresholding is used automatically.
       - The mask is refined via hole filling and morphological closing.
       - Only the largest connected component is kept (assumed to be the brain).
       - Output directory is created if it does not exist.
       - Input file must be a valid 3D NIfTI image.

    :param nii_path:
        Path to the input brain CTA file (``.nii.gz``).

    :param output_path:
        Directory where the brain mask NIfTI file will be saved.

    :param threshold:
        Tuple ``(low, high)`` defining the intensity range used for segmentation.
        If None, Otsu's method determines the threshold automatically.

    :param closing_radius:
        Radius (in voxels) used for morphological closing to smooth the mask.

    :param debug:
        If ``True``, prints diagnostic information about thresholds, data shapes,
        and saved file locations.

    :raises FileNotFoundError:
        If the input NIfTI file does not exist.

    :raises ValueError:
        If the input file is not a 3D image or if the threshold tuple is invalid.

    Example
    -------
    >>> from nidataset.volume import generate_brain_mask
    >>>
    >>> generate_brain_mask(
    ...     nii_path="dataset/CTA_raw/case_01.nii.gz",
    ...     output_path="output/CTA_brain_masks/",
    ...     threshold=None,
    ...     closing_radius=3,
    ...     debug=True
    ... )
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


def generate_brain_mask_dataset(nii_folder: str, 
                                output_path: str, 
                                threshold: tuple = None,
                                closing_radius: int = 3,
                                debug: bool = False) -> None:
    """
    Generates brain masks for all brain CTA scans inside a dataset folder and
    saves each output as:

        <PREFIX>_brain_mask.nii.gz

    .. note::
       - All ``.nii.gz`` files in the folder are processed.
       - Each scan is segmented using ``generate_brain_mask``.
       - If ``threshold`` is None, Otsu’s thresholding is used per-volume.
       - Output directory is created if missing.
       - Input scans must be 3D NIfTI images.

    :param nii_folder:
        Path to the directory containing CTA volumes in ``.nii.gz`` format.

    :param output_path:
        Directory where generated brain masks will be saved.

    :param threshold:
        Tuple ``(low, high)`` specifying the intensity bounds for segmentation.
        If None, automatic Otsu thresholding is applied per scan.

    :param closing_radius:
        Radius used for morphological closing to refine segmentation.

    :param debug:
        If ``True``, prints summary information after processing the dataset.

    :raises FileNotFoundError:
        If the folder does not exist or contains no ``.nii.gz`` files.

    Example
    -------
    >>> from nidataset.volume import generate_brain_mask_dataset
    >>>
    >>> generate_brain_mask_dataset(
    ...     nii_folder="dataset/CTA_raw/",
    ...     output_path="output/CTA_brain_masks/",
    ...     threshold=None,
    ...     closing_radius=3,
    ...     debug=True
    ... )
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
    Finds the minimum bounding box around a CTA scan, crops the volume, pads it 
    to a fixed target size, and preserves spatial orientation. Saves the result as:

        <PREFIX>_cropped_padded.nii.gz

    .. note::
       - Automatically identifies the non-zero bounding box of the volume.
       - Applies centered cropping if the scan exceeds the target size.
       - Applies symmetric padding when the scan is smaller than the target size.
       - Affine and header information are updated to maintain correct orientation.
       - Input scans must be 3D NIfTI images.

    :param nii_path:
        Path to the input CTA volume in ``.nii.gz`` format.

    :param output_path:
        Directory where the cropped and padded scan will be saved.

    :param target_shape:
        Desired output shape ``(X, Y, Z)`` after cropping/padding.

    :param debug:
        If ``True``, prints detailed shape information and the output file path.

    :raises FileNotFoundError:
        If the input file does not exist.

    :raises ValueError:
        If the file is not 3D or contains invalid dimensions.

    Example
    -------
    >>> from nidataset.volume import crop_and_pad
    >>>
    >>> crop_and_pad(
    ...     nii_path="dataset/CTA_raw/case_01.nii.gz",
    ...     output_path="output/CTA_cropped/",
    ...     target_shape=(128, 128, 128),
    ...     debug=True
    ... )
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


def crop_and_pad_dataset(nii_folder: str, 
                         output_path: str, 
                         target_shape: tuple = (128, 128, 128), 
                         save_stats: bool = False) -> None:
    """
    Processes all CTA scans inside a dataset folder, applies ``crop_and_pad`` to each
    volume, and saves every output as:

        <PREFIX>_cropped_padded.nii.gz

    .. note::
       - All ``.nii.gz`` files inside the folder are processed.
       - Each scan is cropped to its non-zero bounding box and padded/cropped to ``target_shape``.
       - Output directory is created automatically if missing.
       - Optionally saves a CSV file with original and final shapes.
       - Input scans must be 3D NIfTI images.

    :param nii_folder:
        Path to the directory containing CTA volumes in ``.nii.gz`` format.

    :param output_path:
        Directory where the cropped and padded scans will be saved.

    :param target_shape:
        Desired output shape ``(X, Y, Z)`` applied uniformly to all scans.

    :param save_stats:
        If ``True``, saves a ``crop_pad_stats.csv`` file containing
        ``FILENAME``, ``ORIGINAL_SHAPE``, and ``FINAL_SHAPE``.

    :raises FileNotFoundError:
        If the folder does not exist or contains no ``.nii.gz`` files.

    Example
    -------
    >>> from nidataset.volume import crop_and_pad_dataset
    >>>
    >>> crop_and_pad_dataset(
    ...     nii_folder="dataset/CTA_raw/",
    ...     output_path="output/CTA_cropped_padded/",
    ...     target_shape=(128, 128, 128),
    ...     save_stats=True
    ... )
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



