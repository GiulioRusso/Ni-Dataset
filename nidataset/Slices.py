import os
from tqdm import tqdm
import numpy as np
import csv
import nibabel as nib
from PIL import Image
from typing import Optional, Tuple


def extract_slices(nii_path: str, 
                   output_path: str, 
                   view: str = "axial",
                   target_size: Optional[Tuple[int, int]] = None,
                   pad_value: float = 0.0,
                   debug: bool = False) -> None:
    """
    Extracts slices from a NIfTI file and saves them as images .tif, following the structure
        <NIFTI FILENAME>_<VIEW>_<PROGRESSIVE SLICE NUMBER>.tif
    
    :param nii_path: 
        Path to the input .nii.gz file with shape (X, Y, Z).
    
    :param output_path: 
        Path where the extracted slices will be saved.
    
    :param view: 
        Anatomical view for slice extraction:
        
        - ``"axial"`` → extracts along the Z-axis.
        - ``"coronal"`` → extracts along the Y-axis.
        - ``"sagittal"`` → extracts along the X-axis.
    
    :param target_size:
        Optional target dimensions (height, width) for the output slices. If specified,
        slices will be padded symmetrically to reach the target size. Padding is applied
        equally on both sides when possible; if odd padding is needed, the extra pixel
        is added to the right/bottom. If ``None``, slices are saved at their original size.
        
        Example: ``target_size=(512, 512)``
    
    :param pad_value:
        Value used for padding when ``target_size`` is specified. Default is ``0.0``.
    
    :param debug: 
        Verbose print about the total number of slices extracted. Default is ``False``.
    
    :raises FileNotFoundError: 
        If the input NIfTI file does not exist.
    
    :raises ValueError: 
        If the NIfTI file is empty, has invalid extension, or invalid dimensions.
        If the view is not 'axial', 'coronal', or 'sagittal'.
        If ``target_size`` is smaller than the original slice dimensions.
    
    Example
    -------
    >>> from nidataset.slices import extract_slices
    >>> 
    >>> # define paths
    >>> nii_path = "path/to/input_image.nii.gz"
    >>> output_path = "path/to/output_directory"
    >>> 
    >>> # choose the anatomical view ('axial', 'coronal', or 'sagittal')
    >>> view = "axial"
    >>> 
    >>> # extract slices without padding
    >>> extract_slices(nii_path=nii_path, 
    ...                output_path=output_path, 
    ...                view=view, 
    ...                debug=True)
    >>> 
    >>> # extract slices with padding to 512x512
    >>> extract_slices(nii_path=nii_path,
    ...                output_path=output_path,
    ...                view=view,
    ...                target_size=(512, 512),
    ...                pad_value=0.0,
    ...                debug=True)
    """
    
    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")
    
    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file. Got '{nii_path}' instead.")
    
    # validate the view parameter
    valid_views = {'axial', 'coronal', 'sagittal'}
    if view not in valid_views:
        raise ValueError(f"Error: The view must be one of {valid_views}. Got '{view}' instead.")
    
    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # load the NIfTI file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    
    # validate NIfTI data dimensions
    if nii_data.ndim != 3:
        raise ValueError(f"Error: expected a 3D NIfTI file. Got shape '{nii_data.shape}' instead.")
    
    # mapping of views to slicing axes
    view_mapping = {
        "axial": (2, lambda data, i: data[:, :, i]),       # Z-axis
        "coronal": (1, lambda data, i: data[:, i, :]),     # Y-axis
        "sagittal": (0, lambda data, i: data[i, :, :])     # X-axis
    }
    
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
        
        # apply padding if target_size is specified
        if target_size is not None:
            target_h, target_w = target_size
            current_h, current_w = slice_data.shape
            
            # validate that target size is not smaller than current size
            if target_h < current_h or target_w < current_w:
                raise ValueError(
                    f"Error: target_size {target_size} is smaller than slice dimensions "
                    f"({current_h}, {current_w}). Target size must be >= slice dimensions."
                )
            
            # calculate padding amounts
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            
            # distribute padding symmetrically (extra pixel goes to right/bottom if odd)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # apply padding
            slice_data = np.pad(
                slice_data,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=pad_value
            )
        
        # construct filename with zero-padded slice index
        slice_filename = f"{prefix}_{view}_{str(i).zfill(3)}.tif"
        slice_path = os.path.join(output_path, slice_filename)
        
        # save slice as an image
        slice_to_save = Image.fromarray(slice_data)
        slice_to_save.save(slice_path)
    
    # debug verbose print
    if debug:
        padding_info = f"\nPadding applied: target size {target_size}" if target_size else "\nNo padding applied"
        print(f"\nInput file: '{nii_path}'\nOutput path: '{output_path}'"
              f"{padding_info}\nTotal {view} slices extracted: {num_slices}")
        

def extract_slices_dataset(nii_folder: str, 
                           output_path: str, 
                           view: str = "axial", 
                           saving_mode: str = "case",
                           target_size: Optional[Tuple[int, int]] = None,
                           pad_value: float = 0.0,
                           save_stats: bool = False) -> None:
    """
    Extracts slices from all NIfTI files in a dataset folder and saves them as images .tif, following the structure

        <NIFTI FILENAME>_<VIEW>_<PROGRESSIVE SLICE NUMBER>.tif

    :param nii_folder: 
        Path to the folder containing all .nii.gz files with shape (X, Y, Z).
    
    :param output_path: 
        Path where the extracted slices will be saved.
    
    :param view: 
        Anatomical view for slice extraction:
        
        - ``"axial"`` → extracts along the Z-axis.
        - ``"coronal"`` → extracts along the Y-axis.
        - ``"sagittal"`` → extracts along the X-axis.
    
    :param saving_mode: 
        - ``"case"`` → creates a folder for each case.
        - ``"view"`` → saves all slices inside a single view folder.
    
    :param target_size:
        Optional target dimensions (height, width) for the output slices. If specified,
        slices will be padded symmetrically to reach the target size. If ``None``, 
        slices are saved at their original size.
        
        Example: ``target_size=(512, 512)``
    
    :param pad_value:
        Value used for padding when ``target_size`` is specified. Default is ``0.0``.
    
    :param save_stats: 
        If ``True``, saves a CSV file with FILENAME and NUM_SLICES information per case 
        as ``<VIEW>_slices_stats.csv``.

    :raises FileNotFoundError: 
        If the dataset folder does not exist or contains no .nii.gz files.
    
    :raises ValueError: 
        If an invalid view or saving_mode is provided.

    Example
    -------
    >>> from nidataset.slices import extract_slices_dataset
    >>> 
    >>> # define paths
    >>> nii_folder = "path/to/dataset"
    >>> output_path = "path/to/output_directory"
    >>> 
    >>> # choose the anatomical view ('axial', 'coronal', or 'sagittal')
    >>> view = "axial"
    >>> 
    >>> # extract slices without padding
    >>> extract_slices_dataset(nii_folder=nii_folder, 
    ...                        output_path=output_path, 
    ...                        view=view, 
    ...                        saving_mode="view",
    ...                        save_stats=True)
    >>> 
    >>> # extract slices with padding to 512x512
    >>> extract_slices_dataset(nii_folder=nii_folder,
    ...                        output_path=output_path,
    ...                        view=view,
    ...                        saving_mode="case",
    ...                        target_size=(512, 512),
    ...                        pad_value=0.0,
    ...                        save_stats=True)
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{nii_folder}' does not exist.")

    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{nii_folder}'.")
    
    # validate the view parameter
    valid_views = {'axial', 'coronal', 'sagittal'}
    if view not in valid_views:
        raise ValueError(f"Error: The view must be one of {valid_views}. Got '{view}' instead.")

    # validate input parameters
    if saving_mode not in ["case", "view"]:
        raise ValueError(f"Error: saving_mode must be either 'case' or 'view'. Got '{saving_mode}' instead.")

    # create output dir if it does not exist
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
            extract_slices(nii_path, case_output_dir, view, 
                          target_size=target_size, pad_value=pad_value, debug=False)
        else:
            extract_slices(nii_path, view_output_dir, view, 
                          target_size=target_size, pad_value=pad_value, debug=False)

    # save statistics if enabled
    if save_stats:
        with open(stats_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["FILENAME", "NUM_SLICES"])
            writer.writerows(stats)
            writer.writerow(["TOTAL_SLICES", total_slices])
        
        padding_info = f" (with padding to {target_size})" if target_size else ""
        print(f"\nSlice extraction statistics saved in: '{stats_file}'{padding_info}")


def extract_annotations(nii_path: str, 
                        output_path: str, 
                        view: str = "axial",
                        saving_mode: str = "slice", 
                        data_mode: str = "center",
                        target_size: Optional[Tuple[int, int]] = None,
                        debug: bool = False) -> None:
    """
    Extracts annotations from a NIfTI annotation file and saves them as CSV, based on the selected view and named with:

        <NIFTI FILENAME>_<VIEW>_<PROGRESSIVE SLICE NUMBER>.csv

    or

        <NIFTI FILENAME>.csv

    :param nii_path: 
        Path to the input .nii.gz file with shape (X, Y, Z).
    
    :param output_path: 
        Path where the CSV annotations will be saved.
    
    :param view: 
        Anatomical view for annotation extraction:
        
        - ``"axial"`` → extracts along the Z-axis.
        - ``"coronal"`` → extracts along the Y-axis.
        - ``"sagittal"`` → extracts along the X-axis.
    
    :param saving_mode: 
        - ``"slice"`` → generates a CSV per slice.
        - ``"volume"`` → generates a single CSV for the whole volume.
    
    :param data_mode: 
        - ``"center"`` → saves the center of the bounding box.
        - ``"box"`` → saves the bounding box coordinates.
        - ``"radius"`` → saves the center and radius (from center to bounding box border).
    
    :param target_size:
        Optional target dimensions (height, width) for coordinate adjustment. If specified,
        coordinates will be adjusted to account for padding that would be applied to match
        the target size. This ensures annotations align with padded images from ``extract_slices``.
        If ``None``, coordinates are saved at their original values. Note that this is applied only if
        the saving_mode is ``"slice"``.
        
        Example: ``target_size=(512, 512)``
    
    :param debug: 
        If ``True``, prints additional information about the extraction.

    :raises FileNotFoundError: 
        If the input NIfTI file does not exist.
    
    :raises ValueError: 
        If the NIfTI file is empty, has invalid extension, or invalid parameters.
        If the view is not 'axial', 'coronal', or 'sagittal'.
        If the saving_mode and data_mode are not correct.

    Example
    -------
    >>> from nidataset.slices import extract_annotations
    >>> 
    >>> # define paths
    >>> nii_path = "path/to/input_image.nii.gz"
    >>> output_path = "path/to/output_directory"
    >>> 
    >>> # choose the anatomical view ('axial', 'coronal', or 'sagittal')
    >>> view = "axial"
    >>> 
    >>> # extract annotations without padding adjustment
    >>> extract_annotations(nii_path=nii_path, 
    ...                     output_path=output_path, 
    ...                     view=view, 
    ...                     saving_mode="slice",
    ...                     data_mode="center",
    ...                     debug=True)
    >>> 
    >>> # extract annotations with padding adjustment to match 512x512 images
    >>> extract_annotations(nii_path=nii_path,
    ...                     output_path=output_path,
    ...                     view=view,
    ...                     saving_mode="slice",
    ...                     data_mode="box",
    ...                     target_size=(512, 512),
    ...                     debug=True)
    >>> 
    >>> # extract annotations with center and radius
    >>> extract_annotations(nii_path=nii_path,
    ...                     output_path=output_path,
    ...                     view=view,
    ...                     saving_mode="volume",
    ...                     data_mode="radius",
    ...                     debug=True)
    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file, but got '{nii_path}'.")

    # validate the view parameter
    valid_views = {'axial', 'coronal', 'sagittal'}
    if view not in valid_views:
        raise ValueError(f"Error: The view must be one of {valid_views}. Got '{view}' instead.")

    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the NIfTI file
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()

    # validate NIfTI data dimensions
    if nii_data.ndim != 3:
        raise ValueError(f"Error: expected a 3D NIfTI file. Got shape '{nii_data.shape}' instead.")

    # validate saving_mode and data_mode
    if saving_mode not in ["slice", "volume"]:
        raise ValueError("Error: saving_mode must be either 'slice' or 'volume'.")
    if data_mode not in ["center", "box", "radius"]:
        raise ValueError("Error: data_mode must be either 'center', 'box', or 'radius'.")

    # extract filename prefix
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")

    # calculate padding offsets if target_size is specified
    pad_offset_x = 0
    pad_offset_y = 0
    
    if target_size is not None:
        target_h, target_w = target_size
        
        # determine slice dimensions based on view
        if view == "axial":
            current_h, current_w = nii_data.shape[0], nii_data.shape[1]
        elif view == "coronal":
            current_h, current_w = nii_data.shape[0], nii_data.shape[2]
        else:  # sagittal
            current_h, current_w = nii_data.shape[1], nii_data.shape[2]
        
        # validate that target size is not smaller than current size
        if target_h < current_h or target_w < current_w:
            raise ValueError(
                f"Error: target_size {target_size} is smaller than slice dimensions "
                f"({current_h}, {current_w}). Target size must be >= slice dimensions."
            )
        
        # calculate padding offsets (matches extract_slices symmetric padding)
        pad_h = target_h - current_h
        pad_w = target_w - current_w
        pad_offset_x = pad_h // 2  # offset for first dimension
        pad_offset_y = pad_w // 2  # offset for second dimension

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

        # adjust coordinates based on view and apply padding offset
        if view == "axial":
            center_x = (min_x + max_x) / 2 + pad_offset_x
            center_y = (min_y + max_y) / 2 + pad_offset_y
            center_z = (min_z + max_z) / 2
            radius_x = (max_x - min_x) / 2
            radius_y = (max_y - min_y) / 2
            radius_z = (max_z - min_z) / 2
            
            if data_mode == "center":
                box_data = [center_x, center_y, min_z, max_z]
            elif data_mode == "radius":
                box_data = [center_x, center_y, center_z, radius_x, radius_y, radius_z]
            else:  # box
                box_data = [min_x + pad_offset_x, min_y + pad_offset_y, min_z, 
                           max_x + pad_offset_x, max_y + pad_offset_y, max_z]
        elif view == "coronal":
            center_x = (min_x + max_x) / 2 + pad_offset_x
            center_y = (min_y + max_y) / 2
            center_z = (min_z + max_z) / 2 + pad_offset_y
            radius_x = (max_x - min_x) / 2
            radius_y = (max_y - min_y) / 2
            radius_z = (max_z - min_z) / 2
            
            if data_mode == "center":
                box_data = [center_x, center_z, min_y, max_y]
            elif data_mode == "radius":
                box_data = [center_x, center_y, center_z, radius_x, radius_y, radius_z]
            else:  # box
                box_data = [min_x + pad_offset_x, min_z + pad_offset_y, min_y,
                           max_x + pad_offset_x, max_z + pad_offset_y, max_y]
        else:  # sagittal
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2 + pad_offset_x
            center_z = (min_z + max_z) / 2 + pad_offset_y
            radius_x = (max_x - min_x) / 2
            radius_y = (max_y - min_y) / 2
            radius_z = (max_z - min_z) / 2
            
            if data_mode == "center":
                box_data = [center_y, center_z, min_x, max_x]
            elif data_mode == "radius":
                box_data = [center_x, center_y, center_z, radius_x, radius_y, radius_z]
            else:  # box
                box_data = [min_y + pad_offset_x, min_z + pad_offset_y, min_x,
                           max_y + pad_offset_x, max_z + pad_offset_y, max_x]

        bounding_boxes.append(box_data)

    # handle extraction mode
    if saving_mode == "volume":
        csv_file = os.path.join(output_path, f"{prefix}.csv")
        with open(csv_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # write appropriate headers based on data_mode
            if data_mode == "center":
                writer.writerow(["CENTER_X", "CENTER_Y", "CENTER_Z"])
                bounding_boxes_vol = [[box[0], box[1], box[2]] for box in bounding_boxes]
            elif data_mode == "radius":
                writer.writerow(["CENTER_X", "CENTER_Y", "CENTER_Z", "RADIUS_X", "RADIUS_Y", "RADIUS_Z"])
                bounding_boxes_vol = bounding_boxes
            else:  # box
                writer.writerow(["X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"])
                bounding_boxes_vol = bounding_boxes

            writer.writerows(bounding_boxes_vol)

        # debug print
        if debug:
            padding_info = f"\nPadding adjustment applied: target size {target_size}" if target_size else "\nNo padding adjustment applied"
            print(f"\nInput file: '{nii_path}'\nOutput path: '{output_path}'"
                  f"{padding_info}\nTotal volume annotations extracted: {len(bounding_boxes)}")

    else:
        slice_annotations = {}

        # build a dict record for each slice with annotation
        for box in bounding_boxes:
            if data_mode == "radius":
                center_z, radius_z = box[2], box[5]
                z_min, z_max = int(center_z - radius_z), int(center_z + radius_z)
            else:
                z_min, z_max = int(box[2]), int(box[3] if data_mode == "center" else box[5])

            for z in range(z_min, z_max + 1):
                if z not in slice_annotations:
                    slice_annotations[z] = []

                if data_mode == "center":
                    # store only X and Y for center mode
                    slice_annotations[z].append([int(box[0]), int(box[1])])
                elif data_mode == "radius":
                    # store center and radii for radius mode
                    slice_annotations[z].append([int(box[0]), int(box[1]), int(box[3]), int(box[4])])
                else:  # box
                    # store full bounding box for box mode
                    slice_annotations[z].append([int(box[0]), int(box[1]), int(box[3]), int(box[4])])

        # process each slice
        num_slices = len(slice_annotations)
        for z, boxes in tqdm(slice_annotations.items(), desc=f"Processing {prefix} (Slices)", unit="slice"):
            slice_filename = f"{prefix}_{view}_{str(z).zfill(3)}.csv"
            slice_file = os.path.join(output_path, slice_filename)

            with open(slice_file, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if data_mode == "center":
                    writer.writerow(["CENTER_X", "CENTER_Y"])
                elif data_mode == "radius":
                    writer.writerow(["CENTER_X", "CENTER_Y", "RADIUS_X", "RADIUS_Y"])
                else:  # box
                    writer.writerow(["X_MIN", "Y_MIN", "X_MAX", "Y_MAX"])
                writer.writerows(boxes)

        # debug print
        if debug:
            padding_info = f"\nPadding adjustment applied: target size {target_size}" if target_size else "\nNo padding adjustment applied"
            print(f"\nInput file: '{nii_path}'\nOutput path: '{output_path}'"
                  f"{padding_info}\nTotal slices with annotations extracted: {num_slices}")
            

def extract_annotations_dataset(nii_folder: str, 
                                output_path: str, 
                                view: str = "axial",
                                saving_mode: str = "case", 
                                extraction_mode: str = "slice", 
                                data_mode: str = "center",
                                target_size: Optional[Tuple[int, int]] = None,
                                save_stats: bool = False) -> None:
    """
    Extracts annotations from all NIfTI annotation files in a dataset folder and saves them as CSV, based on the selected view and named with:

        <NIFTI FILENAME>_<VIEW>_<PROGRESSIVE SLICE NUMBER>.csv

    or

        <NIFTI FILENAME>.csv

    :param nii_folder: 
        Path to the folder containing all .nii.gz files with shape (X, Y, Z).
    
    :param output_path: 
        Path where the extracted annotations will be saved.
    
    :param view: 
        Anatomical view for annotation extraction:
        
        - ``"axial"`` → extracts along the Z-axis.
        - ``"coronal"`` → extracts along the Y-axis.
        - ``"sagittal"`` → extracts along the X-axis.
    
    :param saving_mode: 
        - ``"case"`` → creates a folder for each case.
        - ``"view"`` → saves all CSVs inside a single folder.
    
    :param extraction_mode: 
        - ``"slice"`` → generates a CSV per slice.
        - ``"volume"`` → generates a single CSV for the whole volume.
    
    :param data_mode: 
        - ``"center"`` → saves the center (X, Y, Z) of the bounding box.
        - ``"box"`` → saves the bounding box coordinates.
    
    :param target_size:
        Optional target dimensions (height, width) for coordinate adjustment. If specified,
        coordinates will be adjusted to account for padding. This should match the 
        ``target_size`` used in ``extract_slices_dataset`` to ensure alignment between 
        images and annotations. If ``None``, coordinates are saved at their original values.
        
        Example: ``target_size=(512, 512)``
    
    :param save_stats: 
        If ``True``, saves a CSV file with FILENAME and NUM_ANNOTATIONS information per 
        case as ``<VIEW>_annotations_stats.csv``.

    :raises FileNotFoundError: 
        If the dataset folder does not exist or contains no .nii.gz files.
    
    :raises ValueError: 
        If an invalid view, saving_mode or data_mode is provided.

    Example
    -------
    >>> from nidataset.slices import extract_annotations_dataset
    >>> 
    >>> # define paths
    >>> nii_folder = "path/to/dataset"
    >>> output_path = "path/to/output_directory"
    >>> 
    >>> # choose the anatomical view ('axial', 'coronal', or 'sagittal')
    >>> view = "axial"
    >>> 
    >>> # extract annotations without padding adjustment
    >>> extract_annotations_dataset(nii_folder=nii_folder, 
    ...                             output_path=output_path, 
    ...                             view=view, 
    ...                             saving_mode="view",
    ...                             extraction_mode="slice", 
    ...                             data_mode="center",
    ...                             save_stats=True)
    >>> 
    >>> # extract annotations with padding adjustment to match 512x512 images
    >>> extract_annotations_dataset(nii_folder=nii_folder,
    ...                             output_path=output_path,
    ...                             view=view,
    ...                             saving_mode="case",
    ...                             extraction_mode="slice",
    ...                             data_mode="box",
    ...                             target_size=(512, 512),
    ...                             save_stats=True)
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{nii_folder}' does not exist.")

    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]

    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{nii_folder}'.")

    # validate the view parameter
    valid_views = {'axial', 'coronal', 'sagittal'}
    if view not in valid_views:
        raise ValueError(f"Error: The view must be one of {valid_views}. Got '{view}' instead.")
    
    # validate modes
    if saving_mode not in ["case", "view"]:
        raise ValueError(f"Error: saving_mode must be either 'case' or 'view'. Got '{saving_mode}' instead.")
    if extraction_mode not in ["slice", "volume"]:
        raise ValueError(f"Error: extraction_mode must be either 'slice' or 'volume'. Got '{extraction_mode}' instead.")
    if data_mode not in ["center", "box"]:
        raise ValueError(f"Error: data_mode must be either 'center' or 'box'. Got '{data_mode}' instead.")
    
    # create output dir if it does not exist
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
            tqdm.write(f"Error processing {nii_file} for statistical analysis: {e}")
            continue  # skip this file if an error occurs
        
        # keep track of the total number of annotations
        total_annotations += num_annotations
        if save_stats:
            stats.append([nii_file, num_annotations])

        # determine the appropriate output folder
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix, view)
            os.makedirs(case_output_dir, exist_ok=True)
            extract_annotations(nii_path, case_output_dir, view, extraction_mode, 
                              data_mode, target_size=target_size, debug=False)
        else:
            extract_annotations(nii_path, view_output_dir, view, extraction_mode, 
                              data_mode, target_size=target_size, debug=False)

    # save statistics if enabled
    if save_stats:
        with open(stats_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["FILENAME", "NUM_ANNOTATIONS"])
            writer.writerows(stats)
            writer.writerow(["TOTAL_ANNOTATIONS", total_annotations])
        
        padding_info = f" (with padding adjustment to {target_size})" if target_size else ""
        print(f"\nAnnotation statistics saved in: '{stats_file}'{padding_info}")

