import numpy as np
import nibabel as nib
import torch
import os
import pandas as pd
import cv2
from typing import Tuple



def draw_3D_boxes(df: pd.DataFrame,
                  nii_path: str,
                  output_path: str,
                  intensity_based_on_score: bool = False,
                  debug: bool = False) -> None:
    """
    Draw 3D bounding boxes into a NIfTI volume.

    This function loads a reference ``.nii.gz`` file and overlays 3D bounding boxes
    specified in the input dataframe. The resulting mask is aligned to the spatial
    metadata of the reference volume and saved as:

        <NIFTI_FILENAME>_boxes.nii.gz

    If ``intensity_based_on_score`` is enabled, box intensities are discretized
    into three levels based on the provided ``SCORE`` column. Otherwise, all boxes
    are assigned an intensity of ``1``.

    :param df: 
        Input dataframe containing bounding box coordinates. Required columns:

        - ``X_MIN``, ``Y_MIN``, ``Z_MIN``  
        - ``X_MAX``, ``Y_MAX``, ``Z_MAX``  
        - ``SCORE`` (only if ``intensity_based_on_score=True``)

        Coordinates must be expressed in voxel indices of the reference NIfTI file.

    :param nii_path:
        Path to the reference ``.nii.gz`` image. Metadata (shape, affine) from this
        file are used to ensure correct alignment.

    :param output_path:
        Directory where the output NIfTI file will be written. The directory is
        created if it does not exist.

    :param intensity_based_on_score:
        If ``True``, the bounding box intensity is assigned according to the score:

        - ``score ≤ 0.50`` → intensity ``1``  
        - ``0.50 < score ≤ 0.75`` → intensity ``2``  
        - ``score > 0.75`` → intensity ``3``  

        If ``False``, all boxes are drawn with intensity ``1``.

    :param debug:
        If ``True``, prints detailed information during processing.

    :raises FileNotFoundError:
        If ``nii_path`` does not exist.

    :raises ValueError:
        If required columns are missing or if the dataframe contains NaN values.

    :returns:
        ``None``. A new NIfTI file with the suffix ``_boxes.nii.gz`` is saved to
        ``output_path``.

    Example
    -------
    >>> import pandas as pd
    >>> from nidataset.draw import draw_3D_boxes
    >>> 
    >>> data = {
    ...     'SCORE': [0.3, 0.7, 0.9],
    ...     'X_MIN': [10, 30, 50],
    ...     'Y_MIN': [15, 35, 55],
    ...     'Z_MIN': [20, 40, 60],
    ...     'X_MAX': [20, 40, 60],
    ...     'Y_MAX': [25, 45, 65],
    ...     'Z_MAX': [30, 50, 70],
    ... }
    >>> df = pd.DataFrame(data)
    >>>
    >>> draw_3D_boxes(
    ...     df=df,
    ...     nii_path="path/to/input_image.nii.gz",
    ...     output_path="path/to/output_directory",
    ...     intensity_based_on_score=True,
    ...     debug=True,
    ... )
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
    expected_columns = ['X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX']
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


def draw_2D_annotations(annotation_path: str,
                        image_path: str,
                        output_path: str,
                        radius: int = 5,
                        debug: bool = False) -> None:
    """
    Draw 2D annotations on an image based on slice-format CSV files.

    This function loads an image and overlays annotations specified in the input 
    CSV annotation file. The resulting image is saved as:

        <IMAGE_FILENAME>_annotated.<EXTENSION>

    The annotation file must be in slice format (not volume format) and can contain 
    one of three annotation types: center points, bounding boxes, or radius-based 
    annotations. All drawings are rendered in yellow (BGR: 0, 255, 255).

    :param annotation_path:
        Path to the CSV annotation file. The file must contain one of three formats:

        **Format 1 (Center points):**
        
        - ``CENTER_X``, ``CENTER_Y``
        
        A small circle with radius ``radius`` is drawn at (CENTER_X, CENTER_Y), 
        and a center point is marked.

        **Format 2 (Bounding boxes):**
        
        - ``X_MIN``, ``Y_MIN``, ``X_MAX``, ``Y_MAX``
        
        A rectangle is drawn from (X_MIN, Y_MIN) to (X_MAX, Y_MAX).

        **Format 3 (Radius-based):**
        
        - ``CENTER_X``, ``CENTER_Y``, ``RADIUS_X``, ``RADIUS_Y``
        
        An ellipse is drawn centered at (CENTER_X, CENTER_Y) with axes 
        corresponding to RADIUS_X and RADIUS_Y.

        Coordinates must be expressed in pixel indices of the image.

    :param image_path:
        Path to the input image file. Supported formats include ``.png``, ``.jpg``,
        ``.jpeg``, ``.bmp``, ``.tif``, and other formats supported by OpenCV.

    :param output_path:
        Directory where the output image will be written. The directory is created
        if it does not exist.

    :param radius:
        Radius of the circle drawn around center points (only used for center-based
        annotations without RADIUS_X/RADIUS_Y). Default is ``5``.

    :param debug:
        If ``True``, prints detailed information during processing.

    :raises FileNotFoundError:
        If ``annotation_path`` or ``image_path`` do not exist.

    :raises ValueError:
        If the CSV file does not contain the required columns, contains NaN values,
        or uses volume format (which is not supported by this function).

    :returns:
        ``None``. A new image file with the suffix ``_annotated.<ext>`` is saved to
        ``output_path``.

    Example
    -------
    >>> from nidataset.draw import draw_2D_annotations
    >>> 
    >>> # Example 1: Center-based annotations
    >>> draw_2D_annotations(
    ...     annotation_path="path/to/image_axial_042.csv",
    ...     image_path="path/to/image_axial_042.tif",
    ...     output_path="path/to/output_directory",
    ...     radius=10,
    ...     debug=True,
    ... )
    >>>
    >>> # Example 2: Bounding box annotations
    >>> draw_2D_annotations(
    ...     annotation_path="path/to/image_axial_042.csv",
    ...     image_path="path/to/image_axial_042.tif",
    ...     output_path="path/to/output_directory",
    ...     debug=True,
    ... )
    >>>
    >>> # Example 3: Radius-based annotations
    >>> draw_2D_annotations(
    ...     annotation_path="path/to/image_axial_042.csv",
    ...     image_path="path/to/image_axial_042.tif",
    ...     output_path="path/to/output_directory",
    ...     debug=True,
    ... )
    """

    # check if the annotation file exists
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"Error: the annotation file '{annotation_path}' does not exist.")

    # check if the image file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Error: the image file '{image_path}' does not exist.")

    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the CSV annotation file
    df = pd.read_csv(annotation_path)

    # check if dataframe contains NaN values
    if df.isnull().values.any():
        raise ValueError("Error: The annotation dataframe contains NaN values.")

    # determine annotation format (slice formats only)
    center_columns = ['CENTER_X', 'CENTER_Y']
    bbox_columns = ['X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX']
    radius_columns = ['CENTER_X', 'CENTER_Y', 'RADIUS_X', 'RADIUS_Y']
    
    # check for volume format (not supported)
    volume_indicators = ['CENTER_Z', 'Z_MIN', 'Z_MAX', 'RADIUS_Z']
    if any(col in df.columns for col in volume_indicators):
        raise ValueError(
            "Error: This function only supports slice-format annotations. "
            "Volume-format annotations (with Z coordinates) are not supported."
        )

    is_radius_format = all(col in df.columns for col in radius_columns)
    is_center_format = all(col in df.columns for col in center_columns) and not is_radius_format
    is_bbox_format = all(col in df.columns for col in bbox_columns)

    if not (is_center_format or is_bbox_format or is_radius_format):
        raise ValueError(
            f"Error: The annotation file must contain one of the following formats:\n"
            f"  - Center: {center_columns}\n"
            f"  - Bounding box: {bbox_columns}\n"
            f"  - Radius: {radius_columns}"
        )

    # load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Failed to load image from '{image_path}'.")

    # ensure the image has 3 channels (convert grayscale to BGR if needed)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # define yellow color in BGR
    yellow = (0, 255, 255)
    thickness = 2

    # draw annotations based on format
    if is_center_format:
        for _, row in df.iterrows():
            center_x = int(row['CENTER_X'])
            center_y = int(row['CENTER_Y'])

            # draw circle with specified radius
            cv2.circle(image, (center_x, center_y), radius, yellow, thickness)
            # draw center point
            cv2.circle(image, (center_x, center_y), 1, yellow, -1)

        if debug:
            print(f"Drew {len(df)} center points with radius {radius}.")

    elif is_radius_format:
        for _, row in df.iterrows():
            center_x = int(row['CENTER_X'])
            center_y = int(row['CENTER_Y'])
            radius_x = int(row['RADIUS_X'])
            radius_y = int(row['RADIUS_Y'])

            # draw ellipse with axes corresponding to radii
            cv2.ellipse(image, (center_x, center_y), (radius_x, radius_y), 
                       0, 0, 360, yellow, thickness)
            # draw center point
            cv2.circle(image, (center_x, center_y), 1, yellow, -1)

        if debug:
            print(f"Drew {len(df)} radius-based annotations (ellipses).")

    elif is_bbox_format:
        for _, row in df.iterrows():
            x_min = int(row['X_MIN'])
            y_min = int(row['Y_MIN'])
            x_max = int(row['X_MAX'])
            y_max = int(row['Y_MAX'])

            # draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), yellow, thickness)

        if debug:
            print(f"Drew {len(df)} bounding boxes.")

    # extract filename and extension
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_annotated{ext}"
    output_image_path = os.path.join(output_path, output_filename)

    # save the output image
    cv2.imwrite(output_image_path, image)

    if debug:
        print(f"Annotated image saved at: '{output_image_path}'")


def from_2D_to_3D_coords(df: pd.DataFrame,
                         view: str) -> pd.DataFrame:
    """
    Convert 2D bounding-box or point coordinates into a 3D coordinate system
    based on the specified anatomical view.

    This function interprets 2D coordinates extracted from a particular view
    (``axial``, ``coronal``, or ``sagittal``) and rearranges them into a unified
    3D coordinate convention using the axis order ``(X, Y, Z)``.  
    Both bounding-box style inputs (6 columns) and single-point inputs (3 columns)
    are supported.

    Accepted input formats
    ----------------------
    **Bounding boxes (6 columns)**  
    Required columns:
        - ``X_MIN``, ``Y_MIN``, ``SLICE_NUMBER_MIN``  
        - ``X_MAX``, ``Y_MAX``, ``SLICE_NUMBER_MAX``

    **Points (3 columns)**  
    Required columns:
        - ``X``, ``Y``, ``SLICE_NUMBER``

    After transformation, outputs are always standardized to:

    - Bounding boxes → ``['X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX']``  
    - Points → ``['X', 'Y', 'Z']``  

    :param df:
        Input dataframe containing either 3 or 6 coordinate columns, depending on
        whether the input represents points or bounding boxes.

    :param view:
        Anatomical plane from which the coordinates originate.  
        Must be one of:
        - ``'axial'``  
        - ``'coronal'``  
        - ``'sagittal'``

    :return:
        A new dataframe containing 3D coordinates with axes reordered to match the
        ``(X, Y, Z)`` convention.

    :raises ValueError:
        If the dataframe has an invalid number of columns.  
        If required columns are missing.  
        If ``view`` is not one of the allowed anatomical views.

    Example
    -------
    >>> import pandas as pd
    >>> from nidataset.draw import from_2D_to_3D_coords
    >>>
    >>> data = {
    ...     'X_MIN': [10, 30, 50],
    ...     'Y_MIN': [15, 35, 55],
    ...     'SLICE_NUMBER_MIN': [5, 10, 15],
    ...     'X_MAX': [20, 40, 60],
    ...     'Y_MAX': [25, 45, 65],
    ...     'SLICE_NUMBER_MAX': [10, 15, 20],
    ... }
    >>> df = pd.DataFrame(data)
    >>> df_3d = from_2D_to_3D_coords(df, view='axial')
    >>> print(df_3d)
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
        expected_columns = ['X_MIN', 'Y_MIN', 'SLICE_NUMBER_MIN', 'X_MAX', 'Y_MAX', 'SLICE_NUMBER_MAX']
    else:
        expected_columns = ['X', 'Y', 'SLICE_NUMBER']

    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Error: The input dataframe must contain columns: {expected_columns}")

    # copy the dataframe for modification
    result_df = df.copy()

    # apply coordinate switching based on the anatomical view
    if df.shape[1] == 6:
        if view == 'axial':
            result_df[['X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX']] = df[['Y_MIN', 'X_MIN', 'Y_MAX', 'X_MAX']]
        elif view == 'coronal':
            result_df[['X_MIN', 'Y_MIN', 'SLICE_NUMBER_MIN', 'X_MAX', 'Y_MAX', 'SLICE_NUMBER_MAX']] = df[['SLICE_NUMBER_MIN', 'X_MIN', 'Y_MIN', 'SLICE_NUMBER_MAX', 'X_MAX', 'Y_MAX']]
        elif view == 'sagittal':
            result_df[['X_MIN', 'SLICE_NUMBER_MIN', 'Y_MIN', 'X_MAX', 'SLICE_NUMBER_MAX', 'Y_MAX']] = df[['SLICE_NUMBER_MIN', 'X_MIN', 'Y_MIN', 'SLICE_NUMBER_MAX', 'X_MAX', 'Y_MAX']]
    elif df.shape[1] == 3:
        if view == 'axial':
            result_df[['X', 'Y']] = df[['Y', 'X']]
        elif view == 'coronal':
            result_df[['X', 'Y', 'SLICE_NUMBER']] = df[['SLICE_NUMBER', 'X', 'Y']]
        elif view == 'sagittal':
            result_df[['X', 'SLICE_NUMBER', 'Y']] = df[['SLICE_NUMBER', 'X', 'Y']]

    # rename columns
    if df.shape[1] == 6:
        result_df.columns = ['X_MIN', 'Y_MIN', 'Z_MIN', 'X_MAX', 'Y_MAX', 'Z_MAX']
    else:
        result_df.columns = ['X', 'Y', 'Z']

    return result_df
