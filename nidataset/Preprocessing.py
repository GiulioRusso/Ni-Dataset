import os
from tqdm import tqdm
import subprocess
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter


def skull_CTA(nii_path: str,
              output_path: str,
              f_value: float = 0.1,
              clip_value: tuple = (0, 200),
              cleanup: bool = False,
              debug: bool = False) -> None:
    """
    Perform a CTA-specific skull-stripping pipeline on a single NIfTI file.

    The pipeline applies intensity thresholding, Gaussian smoothing, a second
    thresholding step, and finally FSL BET for skull-stripping. The resulting
    skull-stripped image is intensity-clipped to the specified range.  
    Intermediate images can optionally be removed.

    .. note::
       - The input CTA volume must already be cropped or centered around the
         brain region. ``robust_fov`` is intentionally **not** applied to ensure
         that the input dimensions remain unchanged.
       - This function requires a local FSL installation and access to the
         command-line tools ``fslmaths`` and ``bet``.
       - The script using this function must be executed from a terminal
         (e.g., ``python3 main.py``) so that FSL's environment variables are
         correctly detected.

    :param nii_path:
        Path to the input ``.nii.gz`` file. Must contain a 3D volume of shape
        ``(X, Y, Z)``.

    :param output_path:
        Directory where all intermediate and final outputs will be stored.
        Will be created if it does not exist.

    :param f_value:
        Fractional intensity threshold passed to BET. Typical values range
        from ``0.1`` (more inclusive brain mask) to ``0.3`` (more conservative).

    :param clip_value:
        Tuple ``(min, max)`` defining the intensity range used to clip the
        skull-stripped volume (e.g., ``(0, 200)``).

    :param cleanup:
        If ``True``, removes intermediate files (thresholded and smoothed
        images). The final skull-stripped mask and clipped brain image
        are always preserved.

    :param debug:
        If ``True``, prints detailed information about each processing step.

    :raises FileNotFoundError:
        If ``nii_path`` does not exist.

    :raises ValueError:
        If the file is not a ``.nii.gz`` volume or if the data is not 3D.

    :raises RuntimeError:
        If any FSL command fails.

    Example
    -------
    >>> from nidataset.Processing import skull_CTA
    >>>
    >>> skull_CTA(
    ...     nii_path="patient001_CTA.nii.gz",
    ...     output_path="./processed/",
    ...     f_value=0.2,
    ...     clip_value=(0, 180),
    ...     cleanup=True,
    ...     debug=True
    ... )
    """

    # validate input path
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure data type
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file. Got '{nii_path}' instead.")

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
        print(f"\nSkull-stripped image saved at: '{clipped_img}'\n"
            f"Skull mask saved at: '{mask_img}'")


def skull_CTA_dataset(nii_folder: str,
                      output_path: str,
                      f_value: float = 0.1,
                      clip_value: tuple = (0, 200),
                      cleanup: bool = False,
                      saving_mode: str = "case",
                      debug: bool = False) -> None:
    """
    Apply a CTA-specific skull-stripping pipeline to all NIfTI files inside a folder.

    This function iterates through all ``.nii.gz`` files in the input directory
    and applies the ``skull_CTA`` processing pipeline, which includes
    intensity thresholding, Gaussian smoothing, BET-based skull-stripping,
    and intensity clipping. Processed files can be saved either in a dedicated
    subdirectory per case or all together in a single output folder.

    .. note::
       - All CTA volumes must already be centered on the brain region.
         ``robust_fov`` is intentionally **not** applied to preserve
         the original spatial dimensions.
       - FSL must be installed locally and accessible via command line
         (tools used: ``fslmaths``, ``bet``).
       - When using FSL, scripts must be executed from a terminal to ensure
         correct environment variable detection (e.g., ``python3 main.py``).

    :param nii_folder:
        Directory containing the input ``.nii.gz`` files.

    :param output_path:
        Directory where processed outputs will be saved. Created if missing.

    :param f_value:
        Fractional intensity threshold passed to ``bet`` for skull-stripping.

    :param clip_value:
        Tuple ``(min, max)`` defining the intensity clipping range applied
        to the skull-stripped volume.

    :param cleanup:
        If ``True``, deletes intermediate thresholded and smoothed images.
        The mask and the final clipped CTA image are always retained.

    :param saving_mode:
        Determines how outputs are organized:
        
        - ``"case"`` — creates one subfolder per input file (recommended for datasets).  
        - ``"folder"`` — saves all processed outputs into a single directory.

    :param debug:
        If ``True``, prints detailed information about the skull-stripping
        process for each file.

    :raises FileNotFoundError:
        If ``nii_folder`` does not exist or contains no ``.nii.gz`` files.

    :raises ValueError:
        If ``saving_mode`` is not ``"case"`` or ``"folder"``.

    Example
    -------
    >>> from nidataset.Processing import skull_CTA_dataset
    >>>
    >>> skull_CTA_dataset(
    ...     nii_folder="./CTA_raw/",
    ...     output_path="./CTA_processed/",
    ...     f_value=0.15,
    ...     clip_value=(0, 180),
    ...     cleanup=True,
    ...     saving_mode="case",
    ...     debug=True
    ... )
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{nii_folder}' does not exist.")

    # retrieve all .nii.gz files
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]
    if not nii_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{nii_folder}'.")

    # validate saving_mode
    if saving_mode not in ["case", "folder"]:
        raise ValueError("Error: saving_mode must be either 'case' or 'folder'.")

    # create output dir if it does not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # for "view" mode, create a single folder to store all outputs
    if saving_mode == "folder":
        view_output_dir = os.path.join(output_path)
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

    
def mip(nii_path: str,
        output_path: str,
        window_size: int = 10,
        view: str = "axial",
        debug: bool = False) -> None:
    """
    Generate a sliding-window Maximum Intensity Projection (MIP) from a 3D NIfTI volume.

    For each slice along the chosen anatomical axis, a local neighborhood of size
    ``2 * window_size + 1`` is extracted and collapsed using a max-intensity
    projection. The output is a 3D volume of identical shape to the input, where
    every slice represents a local MIP centered on that slice index.

    The resulting file is saved as:

        ``<FILENAME>_mip_<VIEW>.nii.gz``

    .. note::
       - This is **not** a global projection. Each output slice is generated
         using a local sliding window centered around its index.
       - The input NIfTI must contain a single 3D volume (shape ``(X, Y, Z)``).
       - The affine transformation of the input NIfTI is preserved.

    :param nii_path:
        Path to the input ``.nii.gz`` file. Must contain a 3D CTA/CT volume.

    :param output_path:
        Directory where the MIP output file will be saved.
        Created automatically if it does not exist.

    :param window_size:
        Number of slices on each side of the current slice used to compute
        the local MIP. Effective window length is ``2 * window_size + 1``.

    :param view:
        Anatomical orientation that defines the projection axis:

        - ``"axial"``   → projection along the Z-axis (default)  
        - ``"coronal"`` → projection along the Y-axis  
        - ``"sagittal"``→ projection along the X-axis  

    :param debug:
        If ``True``, prints progress information and the output filename.

    :raises FileNotFoundError:
        If the input file does not exist.

    :raises ValueError:
        If the input file is not ``.nii.gz``, if the NIfTI data is not 3D,
        or if ``view`` is not one of the allowed values.

    Example
    -------
    >>> from nidataset.preprocessing import mip
    >>>
    >>> mip(
    ...     nii_path="CTA_patient001.nii.gz",
    ...     output_path="./MIP_results/",
    ...     window_size=20,
    ...     view="axial",
    ...     debug=True
    ... )
    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")

    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file. Got '{nii_path}' instead.")

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


def mip_dataset(nii_folder: str, 
                output_path: str, 
                window_size: int = 10, 
                view: str = "axial",
                saving_mode: str = "case", 
                debug: bool = False) -> None:
    """
    Generate sliding-window Maximum Intensity Projections (MIP) for all NIfTI
    volumes contained in a dataset directory.

    Each ``.nii.gz`` file is processed independently using the same logic as
    :func:`mip`, producing a local MIP volume with identical shape to the
    original. The output filenames follow the convention:

        ``<FILENAME>_mip_<VIEW>.nii.gz``

    Depending on ``saving_mode``, the output can be organized either into a
    dedicated folder per case or collected into a single view-specific
    directory.

    .. note::
       - Only 3D NIfTI files (shape ``(X, Y, Z)``) are supported.
       - The affine matrix of each input volume is preserved.
       - This function does **not** parallelize processing; files are handled
         sequentially.

    :param nii_folder:
        Path to the dataset directory containing one or more ``.nii.gz`` files.

    :param output_path:
        Directory where the generated MIP files will be saved.
        Created automatically if it does not exist.

    :param window_size:
        Number of slices on each side of the current index used to compute the
        local projection. Effective window length is ``2 * window_size + 1``.

    :param view:
        Anatomical orientation that defines the projection axis:

        - ``"axial"``   → projection along the Z-axis (default)  
        - ``"coronal"`` → projection along the Y-axis  
        - ``"sagittal"``→ projection along the X-axis  

    :param saving_mode:
        Defines how output files are structured:

        - ``"case"`` → creates ``<case>/<view>/`` subfolders (default)
        - ``"view"`` → stores all outputs in a single view-specific directory

    :param debug:
        If ``True``, prints summary information after processing.

    :raises FileNotFoundError:
        If the dataset directory does not exist or contains no ``.nii.gz`` files.

    :raises ValueError:
        If ``view`` or ``saving_mode`` is not one of the allowed values.

    Example
    -------
    >>> from nidataset.preprocessing import mip_dataset
    >>>
    >>> mip_dataset(
    ...     nii_folder="path/to/dataset/",
    ...     output_path="path/to/output/",
    ...     window_size=20,
    ...     view="axial",
    ...     saving_mode="case",
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


def resampling(nii_path: str,
               output_path: str,
               desired_volume: tuple,
               debug: bool = False) -> None:
    """
    Resample a 3D NIfTI volume to a target spatial size while preserving its
    physical field of view.

    The function computes a new voxel spacing such that the original physical
    dimensions of the volume are maintained when interpolating the data into the
    new ``desired_volume`` grid. The output is saved as:

        ``<FILENAME>_resampled.nii.gz``

    .. note::
       - Only 3D NIfTI files (shape ``(X, Y, Z)``) are supported.
       - The affine information (origin, spacing, direction) is recalculated
         consistently using SimpleITK.
       - B-spline interpolation is used for smooth resampling.

    :param nii_path:
        Path to the input ``.nii.gz`` file containing a single 3D volume.

    :param output_path:
        Directory where the resampled volume will be saved. Created if it does
        not exist.

    :param desired_volume:
        Target volume size as a tuple ``(X, Y, Z)``. Must contain exactly three
        integers.

    :param debug:
        If ``True``, prints the location of the saved output.

    :raises FileNotFoundError:
        If the input file does not exist.

    :raises ValueError:
        If the file format is incorrect, the input volume is invalid, or
        ``desired_volume`` does not contain three values.

    Example
    -------
    >>> from nidataset.preprocessing import resampling
    >>>
    >>> resampling(
    ...     nii_path="path/to/input_image.nii.gz",
    ...     output_path="path/to/output/",
    ...     desired_volume=(224, 224, 128),
    ...     debug=True
    ... )
    """

    # check if the input file exists
    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"Error: the input file '{nii_path}' does not exist.")
    
    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file. Got '{nii_path}' instead.")
    
    # ensure tuple has three values
    if len(desired_volume) != 3:
        raise ValueError(f"Error: invalid desired_volume value. Expected three values. Got '{len(desired_volume)}' instead.")

    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # load the NIfTI file
    image = sitk.ReadImage(nii_path)
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    
    # compute new spacing to maintain the same field of view
    new_spacing = original_spacing * (original_size / np.array(desired_volume))
    
    # create resampling filter
    resampled_img = sitk.Resample(
        image,
        desired_volume,
        sitk.Transform(),  # identity transform
        sitk.sitkBSpline,  # smooth interpolation
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID()
    )
    
    # extract filename prefix
    prefix = os.path.basename(nii_path).replace(".nii.gz", "")
    resampled_filename = os.path.join(output_path, f"{prefix}_resampled.nii.gz")
    
    # save the resampled image
    sitk.WriteImage(resampled_img, resampled_filename)
    
    if debug:
        print(f"\nResampled image saved at: '{resampled_filename}'")


def resampling_dataset(nii_folder: str,
                       output_path: str,
                       desired_volume: tuple,
                       saving_mode: str = "case",
                       debug: bool = False) -> None:
    """
    Resample all 3D NIfTI files inside a dataset folder to a target volume size.
    The resampled images preserve the original field of view by computing a new
    voxel spacing consistent with the requested ``desired_volume``. Each output
    file is saved as:

        ``<FILENAME>_resampled.nii.gz``

    .. note::
       - Only 3D ``.nii.gz`` files are processed.
       - Uses B-spline interpolation for smooth volumetric resampling.
       - The output directory structure depends on ``saving_mode``.

    :param nii_folder:
        Directory containing the input ``.nii.gz`` files.

    :param output_path:
        Directory where the resampled images will be saved. Created if it does
        not exist.

    :param desired_volume:
        Target volume size expressed as a tuple ``(X, Y, Z)``. Must contain
        exactly three integers.

    :param saving_mode:
        ``"case"`` → creates a dedicated subfolder for each image  
        ``"folder"`` → saves all resampled images into a single directory

    :param debug:
        If ``True``, prints additional information after processing.

    :raises FileNotFoundError:
        If the input dataset directory does not exist or contains no NIfTI
        files.

    :raises ValueError:
        If ``desired_volume`` has an invalid size or ``saving_mode`` is not
        ``"case"`` or ``"folder"``.

    Example
    -------
    >>> from nidataset.preprocessing import resampling_dataset
    >>>
    >>> resampling_dataset(
    ...     nii_folder="path/to/dataset/",
    ...     output_path="path/to/output/",
    ...     desired_volume=(224, 224, 128),
    ...     saving_mode="case",
    ...     debug=True
    ... )
    """

    # check if the dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{nii_folder}' does not exist.")
    
    # ensure tuple has three values
    if len(desired_volume) != 3:
        raise ValueError(f"Error: invalid desired_volume value. Expected three values. Got '{len(desired_volume)}' instead.")
    
    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]
    
    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{nii_folder}'.")
    
    # validate saving_mode
    if saving_mode not in ["case", "folder"]:
        raise ValueError("Error: saving_mode must be either 'case' or 'folder'.")
    
    # create output dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Processing NIfTI files", unit="file"):
        # nii.gz file path
        nii_path = os.path.join(nii_folder, nii_file)
        
        # extract the filename prefix
        prefix = os.path.basename(nii_path).replace(".nii.gz", "")
        
        # determine the appropriate output folder
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix)
            os.makedirs(case_output_dir, exist_ok=True)
            resampling(nii_path, case_output_dir, desired_volume, debug=False)
        else:
            resampling(nii_path, output_path, desired_volume, debug=False)
    
    if debug:
        print(f"\nResampling completed for all files in '{nii_folder}'")


def register_CTA(nii_path: str,
                 mask_path: str,
                 template_path: str,
                 template_mask_path: str,
                 output_image_path: str,
                 output_transformation_path: str,
                 cleanup: bool = False,
                 debug: bool = False) -> None:
    """
    Registers a CTA volume to a reference template using Mutual Information.
    The pipeline applies Gaussian-based preprocessing on the CTA, loads the
    corresponding masks, performs MI-driven rigid registration, and saves:

        <PREFIX>_registered.nii.gz
        <PREFIX>_gaussian_filtered.nii.gz
        <PREFIX>_transformation.tfm

    .. note::
       - The registration uses a Moment-based initializer, Mattes Mutual 
         Information, and Gradient Descent optimization.
       - The CTA undergoes low/high-intensity suppression and two sequential
         Gaussian smoothings before registration.
       - The template and CTA masks are used to constrain the metric.

    :param nii_path:
        Path to the input CTA ``.nii.gz`` volume.

    :param mask_path:
        Path to the CTA brain mask used to restrict the registration metric.

    :param template_path:
        Path to the reference template image (typically MNI-like CTA template).

    :param template_mask_path:
        Path to the template mask, used as the fixed-image mask.

    :param output_image_path:
        Directory where the registered CTA and temporary filtered CTA will be
        saved. Created if it does not exist.

    :param output_transformation_path:
        Directory where the transformation (``.tfm``) file will be saved.

    :param cleanup:
        If ``True``, deletes the intermediate
        ``<PREFIX>_gaussian_filtered.nii.gz`` file after registration.

    :param debug:
        If ``True``, prints detailed information about the registration process.

    :raises FileNotFoundError:
        If any input file does not exist.

    :raises ValueError:
        If the input file is not a valid ``.nii.gz`` or has invalid dimensions.

    Example
    -------
    >>> from nidataset.preprocessing import register_CTA
    >>>
    >>> register_CTA(
    ...     nii_path="dataset/case001.nii.gz",
    ...     mask_path="dataset/case001_mask.nii.gz",
    ...     template_path="templates/CTA_template.nii.gz",
    ...     template_mask_path="templates/CTA_template_mask.nii.gz",
    ...     output_image_path="output/registered_images/",
    ...     output_transformation_path="output/transforms/",
    ...     cleanup=True,
    ...     debug=True
    ... )
    """

    # check if input files exist
    for file_path in [nii_path, mask_path, template_path, template_mask_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: the input file '{file_path}' does not exist.")
    
    # ensure the file is a .nii.gz file
    if not nii_path.endswith(".nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '.nii.gz' file. Got '{nii_path}' instead.")
    
    # create output directories if they do not exist
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_transformation_path, exist_ok=True)
    
    # extract case number
    prefix = os.path.basename(nii_path).split('-')[0]
    
    # paths for saving outputs
    transformation_path = os.path.join(output_transformation_path, f'{prefix}_transformation.tfm')
    registered_path = os.path.join(output_image_path, f'{prefix}_registered.nii.gz')
    
    # load CTA image
    image = nib.load(nii_path).get_fdata().astype(np.float32)
    
    # apply preprocessing steps
    image[image < 0] = 0  # remove negative values
    image = gaussian_filter(image, sigma=2.0)  # first Gaussian filter
    image[image > 95] = 0  # remove high-intensity values
    image = gaussian_filter(image, sigma=3.0)  # second Gaussian filter
    
    # save preprocessed CTA
    image_gaussian_path = os.path.join(output_image_path, f"{prefix}_gaussian_filtered.nii.gz")
    nib.save(nib.Nifti1Image(image, nib.load(nii_path).affine), image_gaussian_path)
    
    # load images for registration
    image_gaussian = sitk.ReadImage(image_gaussian_path, sitk.sitkFloat32)
    template = sitk.ReadImage(template_path, sitk.sitkFloat32)
    template_mask = sitk.ReadImage(template_mask_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)
    
    # ensure input CTA has the same pixel type as the template
    image_gaussian = sitk.Cast(image_gaussian, template.GetPixelID())
    
    # clip intensity values in CTA (0 to 100)
    image_gaussian = sitk.Clamp(image_gaussian, lowerBound=0, upperBound=100, outputPixelType=image_gaussian.GetPixelID())
    
    # registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # initialize transformation based on image moments
    initial_transform = sitk.CenteredTransformInitializer(
        template_mask, mask, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # set metric as Mutual Information
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricMovingMask(mask)
    registration_method.SetMetricFixedMask(template_mask)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.5)
    
    # interpolation method
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # optimizer settings
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=500, estimateLearningRate=registration_method.Once
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # perform the registration
    transformation = registration_method.Execute(template, image_gaussian)
    
    # save the registered images
    image_registered = sitk.Resample(sitk.ReadImage(nii_path), template, transformation, sitk.sitkLinear, 0.0)
    sitk.WriteImage(image_registered, registered_path)
    
    # save the transformation
    sitk.WriteTransform(transformation, transformation_path)
    
    # delete the temporary gaussian image if cleanup is True
    if cleanup and os.path.exists(image_gaussian_path):
        os.remove(image_gaussian_path)
    
    if debug:
        print(f"\nRegistered image saved at: '{registered_path}'.")
        print(f"Transformation file saved at: '{transformation_path}'.")


def register_CTA_dataset(nii_folder: str,
                         mask_folder: str,
                         template_path: str,
                         template_mask_path: str,
                         output_image_path: str,
                         output_transformation_path: str = "",
                         saving_mode: str = "case",
                         cleanup: bool = False,
                         debug: bool = False) -> None:
    """
    Registers all CTA images in a dataset folder to a reference template using 
    mutual information-based registration. Each CTA volume is preprocessed 
    with Gaussian filtering, masked, and aligned to the template. Saves:

        <PREFIX>_registered.nii.gz
        <PREFIX>_gaussian_filtered.nii.gz
        <PREFIX>_transformation.tfm

    .. note::
       - Each CTA is filtered to remove negative and extreme high-intensity 
         values before registration.
       - The registration uses a moment-based initializer, Mattes Mutual 
         Information metric, and Gradient Descent optimization.
       - The masks constrain the metric to the brain region.
       - If ``saving_mode`` is "case", each case will have its own subfolder 
         containing the registered image and transformation.
       - If ``cleanup`` is True, intermediate Gaussian-filtered images are removed.

    :param nii_folder:
        Path to the folder containing the input CTA ``.nii.gz`` files.

    :param mask_folder:
        Path to the folder containing the corresponding CTA masks. Mask files
        must have the same filenames as the CTA volumes.

    :param template_path:
        Path to the reference template image (CTA volume).

    :param template_mask_path:
        Path to the template mask, used as the fixed-image mask.

    :param output_image_path:
        Directory where registered CTA images will be saved. Created if missing.

    :param output_transformation_path:
        Directory where transformation files (``.tfm``) will be saved. 
        Ignored if ``saving_mode`` is "case" (transform is saved in the case folder).

    :param saving_mode:
        Defines how outputs are organized:

        - ``"case"`` — one subfolder per input file with both registered image 
          and transformation (recommended for datasets).  
        - ``"folder"`` — all registered images saved into a single folder.

    :param cleanup:
        If ``True``, deletes intermediate Gaussian-filtered CTA files.

    :param debug:
        If ``True``, prints detailed information about the registration process
        for each file.

    :raises FileNotFoundError:
        If ``nii_folder`` does not exist or contains no ``.nii.gz`` files.

    :raises ValueError:
        If ``saving_mode`` is not ``"case"`` or ``"folder"``.

    Example
    -------
    >>> from nidataset.preprocessing import register_CTA_dataset
    >>>
    >>> register_CTA_dataset(
    ...     nii_folder="dataset/CTA_raw/",
    ...     mask_folder="dataset/CTA_masks/",
    ...     template_path="templates/CTA_template.nii.gz",
    ...     template_mask_path="templates/CTA_template_mask.nii.gz",
    ...     output_image_path="output/CTA_registered/",
    ...     output_transformation_path="output/CTA_transforms/",
    ...     saving_mode="folder",
    ...     cleanup=True,
    ...     debug=True
    ... )
    """

    # check if dataset folder exists
    if not os.path.isdir(nii_folder):
        raise FileNotFoundError(f"Error: the dataset folder '{nii_folder}' does not exist.")
    
    # get all .nii.gz files in the dataset folder
    nii_files = [f for f in os.listdir(nii_folder) if f.endswith(".nii.gz")]
    
    # check if there are NIfTI files in the dataset folder
    if not nii_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{nii_folder}'.")
    
    # validate saving_mode
    if saving_mode not in ["case", "folder"]:
        raise ValueError("Error: saving_mode must be either 'case' or 'folder'.")
    
    # create output directories if they do not exist
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_transformation_path, exist_ok=True)
    
    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Processing CTA files", unit="file"):
        # paths for input files
        nii_path = os.path.join(nii_folder, nii_file)
        mask_path = os.path.join(mask_folder, nii_file)
        
        # extract the filename prefix
        prefix = os.path.basename(nii_file).replace(".nii.gz", "")
        
        # determine the appropriate output folder
        if saving_mode == "case":
            case_output_image_dir = os.path.join(output_image_path, prefix)
            case_output_transformation_dir = case_output_image_dir
            os.makedirs(case_output_image_dir, exist_ok=True)
            
            register_CTA(nii_path, mask_path, template_path, template_mask_path,
                         case_output_image_dir, case_output_transformation_dir, cleanup, debug)
        else:
            register_CTA(nii_path, mask_path, template_path, template_mask_path,
                         output_image_path, output_transformation_path, cleanup, debug)
    
    if debug:
        print(f"\nRegistration completed for all files in '{nii_folder}'.")

