import numpy as np
import nibabel as nib
import torch
import os


def draw_boxes_on_nifti(tensor: torch.Tensor,
                        nii_path: str,
                        output_path: str,
                        intensity_based_on_score: bool = False,
                        debug: bool = False) -> None:
    """
    Draws 3D bounding boxes on a nii.gz file based on the provided tensor and saves a new nii.gz file inside the specified output path.

    :param tensor: A tensor containing columns with the following structure: ['SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'] where XYZ are already in the 3D reference system.
    :param nii_path: Path to the original nii.gz file.
    :param output_path: Output path.
    :param intensity_based_on_score: If True, use the 'SCORE' column for box intensity with steps. Otherwise, use intensity 1.
    :param debug: if True, prints additional information about the draw.

    :raises FileNotFoundError: if the dataset folder does not exist or contains no .nii.gz files.
    :raises ValueError: if an invalid view or saving_mode is provided.
    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file, but got '{nii_path}'.")

    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # load the nii.gz file
    nifti_image = nib.load(nii_path)
    affine = nifti_image.affine

    # create a new data array for output
    x_axis, y_axis, z_axis = nifti_image.shape
    output_data = np.zeros((x_axis, y_axis, z_axis))

    # process each row in the tensor to draw boxes
    for _, row in enumerate(tensor):
        score, x_min, y_min, z_min, x_max, y_max, z_max = row.tolist()

        # determine the intensity for the box based on the score
        if intensity_based_on_score:
            if score <= 0.5:
                intensity = 1
            elif score <= 0.75:
                intensity = 2
            else:
                intensity = 3
        else:
            intensity = 1

        # draw the box
        output_data[int(x_min):int(x_max), int(y_min):int(y_max), int(z_min):int(z_max),] = intensity

    # create a new Nifti image
    new_nifti_image = nib.Nifti1Image(output_data, affine)
    new_nifti_filename = os.path.basename(nii_path).replace('.nii.gz', '_with_boxes.nii.gz')
    nii_output_path =  os.path.join(output_path, new_nifti_filename)
    nib.save(new_nifti_image, nii_output_path)

    if debug:
        print(f"New nii.gz file saved at: {nii_output_path}")


def from_2D_to_3D_coords(tensor: torch.Tensor,
                         view: str) -> torch.Tensor:
    """
    Switches the box coordinates in the tensor based on the specified anatomical view.

    :param tensor: A tensor containing columns with the following structure: ['X MIN', 'Y MIN', 'SLICE NUMBER MIN', 'X MAX', 'Y MAX', 'SLICE NUMBER MAX'] or ['X', 'Y', 'SLICE NUMBER'].
    :param view: The view from which to adjust the coordinates ('axial', 'coronal', 'sagittal').

    :return result: The tensor with adjusted coordinates.

    :raises ValueError: if an invalid number of columns is provided inside the input tensor.
    """

    if tensor.shape[1] not in (3, 6):
        raise ValueError(f"Error: The input tensor has to be with 3 or 6 columns. Got {tensor.shape[1]} instead.")

    # create a copy of the tensor to modify
    result = tensor.clone()

    if result.shape[1] == 6:
        if 'axial' in view:
            # switch X and Y coordinates
            result[0, 1, 2, 3, 4, 5] = result[1, 0, 2, 4, 3, 5]
        elif 'coronal' in view:
            # switch X with Y and Y with SLICE NUMBER
            result[0, 1, 2, 3, 4, 5] = result[2, 0, 1, 5, 3, 4]
        elif 'sagittal' in view:
            # switch X with SLICE NUMBER
            result[0, 1, 2, 3, 4, 5] = result[2, 3, 0, 5, 4, 3]
    elif result.shape[1] == 3:
        if 'axial' in view:
            # switch X and Y coordinates
            result[0, 1, 2] = result[1, 0, 2]
        elif 'coronal' in view:
            # switch X with Y and Y with SLICE NUMBER
            result[0, 1, 2] = result[2, 0, 1]
        elif 'sagittal' in view:
            # switch X with SLICE NUMBER
            result[0, 1, 2] = result[2, 3, 0]

    return result


