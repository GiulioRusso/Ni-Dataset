import numpy as np
import nibabel as nib
import torch
import os
import pandas as pd


def draw_boxes(df: pd.DataFrame,
               nii_path: str,
               output_path: str,
               intensity_based_on_score: bool = False,
               debug: bool = False) -> None:
    """
    Draws 3D bounding boxes on a nii.gz file. The coords have to be specified inside the input dataframe. A Nifti file is requested as reference and to draw the boxes aligned to this file.
    The file is saved as:

        <NIFTI FILENAME>_boxes.nii.gz
    
    :param df: a dataframe containing columns: ['SCORE', 'X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX'] belonging to a 3D reference system.
    :param nii_path: path to the original nii.gz file from which metadata are taken (in order to keep alignment)
    :param output_path: output path where the nifti draw will be saved.
    :param intensity_based_on_score: if True, use the 'SCORE' column for box intensity with steps. Otherwise, use intensity 1.
    :param debug: if True, prints additional information about the draw.

    :raises FileNotFoundError: if the input NIfTI file does not exist.
    :raises ValueError: if the input dataframe does not contain the required columns.
    :raises ValueError: if the input dataframe contains NaN values.

    
    Example:

        import pandas as pd
        from nidataset.Draw import draw_boxes_on_nifti

        # example dataframe with bounding boxes
        data = {
            'SCORE': [0.3, 0.7, 0.9],
            'X MIN': [10, 30, 50],
            'Y MIN': [15, 35, 55],
            'Z MIN': [20, 40, 60],
            'X MAX': [20, 40, 60],
            'Y MAX': [25, 45, 65],
            'Z MAX': [30, 50, 70]
        }

        df = pd.DataFrame(data)

        # specify input and output paths
        nii_path = "path/to/input_image.nii.gz"
        output_path = "path/to/output_directory"

        # call the function
        draw_boxes(df=df,
                   nii_path=nii_path,
                   output_path=output_path,
                   intensity_based_on_score=True,
                   debug=True)
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

    # define expected columns based on intensity_based_on_score flag
    expected_columns = ['X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX']
    if intensity_based_on_score:
        expected_columns.insert(0, 'SCORE')
    
    # validate dataframe
    if not all(col in df.columns for col in expected_columns):
         raise ValueError(f"Error: The input dataframe must contain columns: {expected_columns}")
    
    # load the nii.gz file
    nifti_image = nib.load(nii_path)
    affine = nifti_image.affine

    # create a new data array for output
    x_axis, y_axis, z_axis = nifti_image.shape
    output_data = np.zeros((x_axis, y_axis, z_axis))

    # process each row in the tensor to draw boxes
    for _, row in df.iterrows():
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

    # extract filename prefix
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")

    # create a new Nifti image
    nifti_draw = nib.Nifti1Image(output_data, affine)
    nii_output_path =  os.path.join(output_path, f"{prefix}_boxes.nii.gz")
    nib.save(nifti_draw, nii_output_path)

    if debug:
        print(f"Boxes draw saved at: '{nii_output_path}'")


def from_2D_to_3D_coords(df: pd.DataFrame,
                         view: str) -> pd.DataFrame:
    """
    Switches the box coordinates in the dataframe based on the specified anatomical view.

    :param df: a dataframe containing columns with the following structure: 
               ['X MIN', 'Y MIN', 'SLICE NUMBER MIN', 'X MAX', 'Y MAX', 'SLICE NUMBER MAX'] 
               or ['X', 'Y', 'SLICE NUMBER'].
    :param view: the view from which to adjust the coordinates ('axial', 'coronal', 'sagittal').

    :return result_df: the dataframe with adjusted coordinates.

    :raises ValueError: if an invalid number of columns is provided inside the input dataframe.
    :raises ValueError: if the dataframe is missing required columns.
    :raises ValueError: if the view is not 'axial', 'coronal', or 'sagittal'.

    Example:

        import pandas as pd
        from nidataset.Draw import from_2D_to_3D_coords

        # example dataframe with bounding box coordinates in 2D (axial view)
        data = {
            'X MIN': [10, 30, 50],
            'Y MIN': [15, 35, 55],
            'SLICE NUMBER MIN': [5, 10, 15],
            'X MAX': [20, 40, 60],
            'Y MAX': [25, 45, 65],
            'SLICE NUMBER MAX': [10, 15, 20]
        }

        df = pd.DataFrame(data)

        # specify the anatomical view
        view = 'axial'

        # convert 2D coordinates to 3D
        df_3d = from_2D_to_3D_coords(df, view)

        # display the modified dataframe
        print(df_3d)

    """

    # validate the view parameter
    valid_views = {'axial', 'coronal', 'sagittal'}
    if view not in valid_views:
        raise ValueError(f"Error: The view must be one of {valid_views}. Got '{view}'.")

    # validate the number of columns
    if df.shape[1] not in (3, 6):
        raise ValueError(f"Error: The input dataframe must have 3 or 6 columns. Got '{df.shape[1]}'.")

    # validate the column names
    if df.shape[1] == 6:
        expected_columns = ['X MIN', 'Y MIN', 'SLICE NUMBER MIN', 'X MAX', 'Y MAX', 'SLICE NUMBER MAX']
    else:
        expected_columns = ['X', 'Y', 'SLICE NUMBER']

    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Error: The input dataframe must contain columns: {expected_columns}")

    # copy the dataframe for modification
    result_df = df.copy()

    # apply coordinate switching based on the anatomical view
    if df.shape[1] == 6:
        if view == 'axial':
            result_df[['X MIN', 'Y MIN', 'X MAX', 'Y MAX']] = df[['Y MIN', 'X MIN', 'Y MAX', 'X MAX']]
        elif view == 'coronal':
            result_df[['X MIN', 'Y MIN', 'SLICE NUMBER MIN', 'X MAX', 'Y MAX', 'SLICE NUMBER MAX']] = df[['SLICE NUMBER MIN', 'X MIN', 'Y MIN', 'SLICE NUMBER MAX', 'X MAX', 'Y MAX']]
        elif view == 'sagittal':
            result_df[['X MIN', 'SLICE NUMBER MIN', 'Y MIN', 'X MAX', 'SLICE NUMBER MAX', 'Y MAX']] = df[['SLICE NUMBER MIN', 'X MIN', 'Y MIN', 'SLICE NUMBER MAX', 'X MAX', 'Y MAX']]
    elif df.shape[1] == 3:
        if view == 'axial':
            result_df[['X', 'Y']] = df[['Y', 'X']]
        elif view == 'coronal':
            result_df[['X', 'Y', 'SLICE NUMBER']] = df[['SLICE NUMBER', 'X', 'Y']]
        elif view == 'sagittal':
            result_df[['X', 'SLICE NUMBER', 'Y']] = df[['SLICE NUMBER', 'X', 'Y']]

    # rename columns
    if df.shape[1] == 6:
        result_df.columns = ['X MIN', 'Y MIN', 'Z MIN', 'X MAX', 'Y MAX', 'Z MAX']
    else:
        result_df.columns = ['X', 'Y', 'Z']

    return result_df
