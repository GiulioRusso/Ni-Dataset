import numpy as np
import nibabel as nib
import torch


def draw_boxes_on_nifti(tensor: torch.Tensor,
                        nifti_file_path: str,
                        intensity_based_on_score: bool = False,
                        debug: bool = False) -> None:
    """
    Draws 3D bounding boxes on a nii.gz file based on the provided tensor and saves a new nii.gz file.

    :param tensor: A tensor containing columns ['SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'] where XYZ are already in the 3D reference system.
    :param nifti_file_path: Path to the original nii.gz file.
    :param intensity_based_on_score: If True, use the 'SCORE' column for box intensity with steps. Otherwise, use intensity 1.
    :param debug: if True, prints additional information about the draw.

    Returns:
    None, saves a new nii.gz file with drawn boxes.
    """
    
    # load the nii.gz file
    nifti_image = nib.load(nifti_file_path)
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
    new_file_path = nifti_file_path.replace('.nii.gz', '_with_boxes.nii.gz')
    nib.save(new_nifti_image, new_file_path)

    if debug:
        print(f"New nii.gz file saved at: {new_file_path}")


def from_2D_to_3D_coords(tensor: torch.Tensor, view: str) -> torch.Tensor:
    """
    Switches the box coordinates in the tensor based on the specified anatomical view.

    Args:
    tensor (torch.Tensor): A tensor with columns ['X MIN', 'Y MIN', 'SLICE NUMBER MIN', 'X MAX', 'Y MAX', 'SLICE NUMBER MAX'] or ['X', 'Y', 'SLICE NUMBER'].
    view (str): The view from which to adjust the coordinates ('axial', 'coronal', 'sagittal').

    Returns:
    torch.Tensor: The tensor with adjusted coordinates.
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


