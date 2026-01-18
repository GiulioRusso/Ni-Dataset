import os
from tqdm import tqdm
import subprocess
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import tempfile
import shutil


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

    # create a temporary directory without spaces for FSL processing
    temp_dir = tempfile.mkdtemp(prefix="fsl_", dir="/tmp")

    try:
        # build intermediate paths in temp directory (no spaces!)
        base_name = os.path.basename(nii_path).replace(".nii.gz", "")
        th_img = os.path.join(temp_dir, f"{base_name}_th.nii.gz")
        th_sm_img = os.path.join(temp_dir, f"{base_name}_th_sm.nii.gz")
        th_sm_th_img = os.path.join(temp_dir, f"{base_name}_th_sm_th.nii.gz")
        skulled_img = os.path.join(temp_dir, f"{base_name}_skulled.nii.gz")
        mask_img = os.path.join(temp_dir, f"{base_name}_skulled_mask.nii.gz")
        brain_img = os.path.join(temp_dir, f"{base_name}_brain.nii.gz")

        # final output paths (in original output_path)
        final_mask_img = os.path.join(output_path, f"{base_name}_skulled_mask.nii.gz")
        final_clipped_img = os.path.join(output_path, f"{base_name}_skulled_clipped.nii.gz")

        if debug:
            print(f"  - Using temporary directory: {temp_dir}")

        # step 2: intensity clipping and smoothing
        subprocess.run(["fslmaths", nii_path, "-thr", "0", "-uthr", "100", th_img],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if debug:
            print(f"  - Threshold [0-100]")

        subprocess.run(["fslmaths", th_img, "-s", "1", th_sm_img],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if debug:
            print(f"  - Smoothing (sigma=1)")

        subprocess.run(["fslmaths", th_sm_img, "-thr", "0", "-uthr", "100", th_sm_th_img],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if debug:
            print("Step 2: Intensity thresholding and smoothing complete.")

        # step 3: BET skull stripping (creates skulled image + mask)
        if debug:
            print(f"Step 3: Running BET with f={f_value}...")

        result = subprocess.run([
            "bet", th_sm_th_img, skulled_img, "-R",
            "-f", str(f_value), "-g", "0", "-m"
        ], check=True, capture_output=True, text=True)

        if debug:
            if result.stdout:
                print(f"  BET stdout: {result.stdout}")
            if result.stderr:
                print(f"  BET stderr: {result.stderr}")
            print(f"Step 3: BET skull stripping complete.")

        # verify BET outputs
        if not os.path.exists(mask_img):
            raise RuntimeError(f"BET failed to create mask: {mask_img}")
        if not os.path.exists(skulled_img):
            raise RuntimeError(f"BET failed to create skulled image: {skulled_img}")

        # step 4: apply brain mask to the ORIGINAL CTA
        subprocess.run(["fslmaths", nii_path, "-mul", mask_img, brain_img],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if debug:
            print("Step 4: Brain mask applied to original CTA.")

        # step 5: clip the final brain image
        nii_img = nib.load(brain_img)
        nii_data = nii_img.get_fdata()
        clipped_data = np.clip(nii_data, clip_value[0], clip_value[1])
        clipped_nii = nib.Nifti1Image(clipped_data, nii_img.affine, nii_img.header)
        nib.save(clipped_nii, final_clipped_img)

        if debug:
            print(f"Step 5: Final brain image clipped to range {clip_value}.")

        # copy mask to final output location
        shutil.copy2(mask_img, final_mask_img)

        if debug:
            print(f"\nSkull-stripped image saved at: '{final_clipped_img}'")
            print(f"Skull mask saved at: '{final_mask_img}'")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else ""
        if not error_msg and hasattr(e, 'stdout') and e.stdout:
            error_msg = e.stdout
        if not error_msg:
            error_msg = f"Command returned exit status {e.returncode}"

        raise RuntimeError(f"FSL command failed for '{nii_path}' with error:\n{error_msg}")

    finally:
        # always cleanup temporary directory
        if os.path.exists(temp_dir):
            if not cleanup:
                # if user wants to keep intermediates, move them to output_path
                for file in os.listdir(temp_dir):
                    src = os.path.join(temp_dir, file)
                    dst = os.path.join(output_path, file)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)

            # remove temp directory
            shutil.rmtree(temp_dir)
            if debug and cleanup:
                print("Temporary files removed.")


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
    subdirectory per case or organized into separate folders for images and masks.

    .. note::
       - All CTA volumes must already be centered on the brain region.
         ``robust_fov`` is intentionally **not** applied to preserve
         the original spatial dimensions.
       - FSL must be installed locally and accessible via command line
         (tools used: ``fslmaths``, ``bet``).
       - When using FSL, scripts must be executed from a terminal to ensure
         correct environment variable detection (e.g., ``python3 main.py``).
       - In folder mode, temporary directories are used during processing and
         are preserved if ``cleanup=False``.

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
        In folder mode, also deletes temporary directories.
        The mask and the final clipped CTA image are always retained.

    :param saving_mode:
        Determines how outputs are organized:
        
        - ``"case"`` — creates one subfolder per input file (recommended for datasets).  
        - ``"folder"`` — saves all skull-stripped images into ``skulled/`` subfolder 
          and all masks into ``masks/`` subfolder. Temporary directories are kept
          if ``cleanup=False``.

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
    os.makedirs(output_path, exist_ok=True)

    # for "folder" mode, create subdirectories for skulled images and masks
    if saving_mode == "folder":
        skulled_dir = os.path.join(output_path, "skulled")
        masks_dir = os.path.join(output_path, "masks")
        os.makedirs(skulled_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

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

        else:  # saving_mode = "folder"
            # for folder mode, process to a temporary location
            temp_output_dir = os.path.join(output_path, f"_temp_{prefix}")
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # process with cleanup=False to keep all files in temp directory
            skull_CTA(
                nii_path=nii_path,
                output_path=temp_output_dir,
                f_value=f_value,
                clip_value=clip_value,
                cleanup=False,
                debug=debug
            )
            
            # move skull-stripped image to skulled folder
            src_skulled = os.path.join(temp_output_dir, f"{prefix}_skulled_clipped.nii.gz")
            dst_skulled = os.path.join(skulled_dir, f"{prefix}_skulled_clipped.nii.gz")
            if os.path.exists(src_skulled):
                os.rename(src_skulled, dst_skulled)
            
            # move mask to masks folder
            src_mask = os.path.join(temp_output_dir, f"{prefix}_skulled_mask.nii.gz")
            dst_mask = os.path.join(masks_dir, f"{prefix}_skulled_mask.nii.gz")
            if os.path.exists(src_mask):
                os.rename(src_mask, dst_mask)
            
            # if cleanup is True, remove the temporary directory
            if cleanup and os.path.exists(temp_output_dir):
                # remove all files in temp directory
                for file in os.listdir(temp_output_dir):
                    os.remove(os.path.join(temp_output_dir, file))
                os.rmdir(temp_output_dir)

    if debug:
        print(f"\nSkull-stripping completed for all files in '{nii_folder}'.")

    
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

        ``<PREFIX>_mip_<VIEW>.nii.gz``

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

        ``<PREFIX>_mip_<VIEW>.nii.gz``

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

        ``<PREFIX>_resampled.nii.gz``

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
    # get original intensity range
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    orig_min = stats.GetMinimum()
    orig_max = stats.GetMaximum()

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

    # clip resampled values to original intensity range
    resampled_img = sitk.Clamp(
        resampled_img,
        lowerBound=orig_min,
        upperBound=orig_max
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

        ``<PREFIX>_resampled.nii.gz``

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
                 output_path: str,
                 cleanup: bool = False,
                 debug: bool = False,
                 number_histogram_bins: int = 128,
                 learning_rate: float = 0.0001,
                 number_iterations: int = 2000,
                 initialization_strategy: int = sitk.CenteredTransformInitializerFilter.MOMENTS,
                 sigma_first: float = 2.0,
                 sigma_second: float = 3.0,
                 metric_sampling_percentage: float = 0.5,
                 initial_transform = None) -> None:
    """
    Registers a CTA volume to a reference template using Mutual Information.
    The pipeline applies Gaussian-based preprocessing on the CTA, loads the
    corresponding masks, performs MI-driven rigid registration, and saves:

        <PREFIX>_registered.nii.gz
        <PREFIX>_gaussian_filtered.nii.gz
        <PREFIX>_transformation.tfm

    .. note::
       - The registration uses a configurable initializer (MOMENTS or GEOMETRY), 
         Mattes Mutual Information, and Gradient Descent optimization.
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

    :param output_path:
        Directory where all output files will be saved (registered CTA, 
        transformation, and temporary filtered CTA). Created if it does not exist.

    :param cleanup:
        If ``True``, deletes the intermediate
        ``<PREFIX>_gaussian_filtered.nii.gz`` file after registration.

    :param debug:
        If ``True``, prints detailed information about the registration process.

    :param number_histogram_bins:
        Number of histogram bins for Mattes Mutual Information metric.
        Default: 128. Common values: 10, 50, 64, 128.

    :param learning_rate:
        Learning rate for Gradient Descent optimizer.
        Default: 0.0001. Common values: 0.0001-1.0.

    :param number_iterations:
        Maximum number of optimization iterations.
        Default: 2000. Common values: 500-5000.

    :param initialization_strategy:
        Strategy for initializing the transformation. Options:
        
        - ``sitk.CenteredTransformInitializerFilter.MOMENTS`` (default) — 
          align based on image moments (center of mass)
        - ``sitk.CenteredTransformInitializerFilter.GEOMETRY`` — 
          align based on image geometry (center and orientation)

    :param sigma_first:
        Standard deviation for the first Gaussian smoothing filter.
        Default: 2.0.

    :param sigma_second:
        Standard deviation for the second Gaussian smoothing filter.
        Default: 3.0.

    :param metric_sampling_percentage:
        Percentage of voxels to sample for metric evaluation (0.0-1.0).
        Default: 0.5 (50%).

    :param initial_transform:
        Initial transformation object. If ``None``, defaults to 
        ``sitk.Euler3DTransform()``. Can be any SimpleITK transform type
        (e.g., ``sitk.Euler3DTransform()``, ``sitk.AffineTransform(3)``).

    :raises FileNotFoundError:
        If any input file does not exist.

    :raises ValueError:
        If the input file is not a valid ``.nii.gz`` or has invalid dimensions.

    Example
    -------
    >>> from nidataset.preprocessing import register_CTA
    >>> import SimpleITK as sitk
    >>>
    >>> # basic usage with default parameters
    >>> register_CTA(
    ...     nii_path="dataset/case001.nii.gz",
    ...     mask_path="dataset/case001_mask.nii.gz",
    ...     template_path="templates/CTA_template.nii.gz",
    ...     template_mask_path="templates/CTA_template_mask.nii.gz",
    ...     output_path="output/case001/",
    ...     cleanup=True
    ... )
    >>>
    >>> # custom registration parameters
    >>> register_CTA(
    ...     nii_path="dataset/case001.nii.gz",
    ...     mask_path="dataset/case001_mask.nii.gz",
    ...     template_path="templates/CTA_template.nii.gz",
    ...     template_mask_path="templates/CTA_template_mask.nii.gz",
    ...     output_path="output/case001/",
    ...     number_histogram_bins=64,
    ...     learning_rate=0.01,
    ...     number_iterations=1000,
    ...     initialization_strategy=sitk.CenteredTransformInitializerFilter.GEOMETRY,
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
    
    # create output directory if it does not exist
    os.makedirs(output_path, exist_ok=True)
    
    # extract case number
    prefix = os.path.basename(nii_path).split('.nii.gz')[0]
    
    # paths for saving outputs
    transformation_path = os.path.join(output_path, f'{prefix}_transformation.tfm')
    registered_folder = os.path.join(output_path, f'{prefix}_registered.nii.gz')
    
    # load CTA image
    image = nib.load(nii_path).get_fdata().astype(np.float32)
    
    # apply preprocessing steps
    image[image < 0] = 0  # remove negative values
    image = gaussian_filter(image, sigma=sigma_first)  # first Gaussian filter
    image[image > 95] = 0  # remove high-intensity values
    image = gaussian_filter(image, sigma=sigma_second)  # second Gaussian filter
    
    # save preprocessed CTA
    image_gaussian_path = os.path.join(output_path, f"{prefix}_gaussian_filtered.nii.gz")
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
    
    # set initial transform type (default to Euler3DTransform if None)
    if initial_transform is None:
        initial_transform = sitk.Euler3DTransform()
    
    # initialize transformation based on selected strategy
    initial_transform = sitk.CenteredTransformInitializer(
        template_mask, mask, initial_transform, initialization_strategy
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # set metric as Mutual Information
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_histogram_bins)
    registration_method.SetMetricMovingMask(mask)
    registration_method.SetMetricFixedMask(template_mask)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(metric_sampling_percentage)
    
    # interpolation method
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # optimizer settings
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=learning_rate, 
        numberOfIterations=number_iterations, 
        estimateLearningRate=registration_method.Once
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # perform the registration
    transformation = registration_method.Execute(template, image_gaussian)
    
    # save the registered images
    image_registered = sitk.Resample(sitk.ReadImage(nii_path), template, transformation, sitk.sitkLinear, 0.0)
    sitk.WriteImage(image_registered, registered_folder)
    
    # save the transformation
    sitk.WriteTransform(transformation, transformation_path)
    
    # delete the temporary gaussian image if cleanup is True
    if cleanup and os.path.exists(image_gaussian_path):
        os.remove(image_gaussian_path)
    
    if debug:
        print(f"\nRegistered image saved at: '{registered_folder}'.")
        print(f"Transformation file saved at: '{transformation_path}'.")


def register_CTA_dataset(nii_folder: str,
                         mask_folder: str,
                         template_path: str,
                         template_mask_path: str,
                         output_path: str,
                         saving_mode: str = "case",
                         cleanup: bool = False,
                         debug: bool = False,
                         number_histogram_bins: int = 128,
                         learning_rate: float = 0.0001,
                         number_iterations: int = 2000,
                         initialization_strategy: int = sitk.CenteredTransformInitializerFilter.MOMENTS,
                         sigma_first: float = 2.0,
                         sigma_second: float = 3.0,
                         metric_sampling_percentage: float = 0.5,
                         initial_transform = None) -> None:
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
       - The registration uses a configurable initializer (MOMENTS or GEOMETRY), 
         Mattes Mutual Information metric, and Gradient Descent optimization.
       - The masks constrain the metric to the brain region.
       - If ``saving_mode`` is "case", each case will have its own subfolder 
         containing the registered image and transformation.
       - If ``saving_mode`` is "folder", all registered images will be saved in 
         ``output_path/registered/`` and all transformations in 
         ``output_path/transforms/``.
       - If ``cleanup`` is True, intermediate Gaussian-filtered images are removed.
       - In folder mode with ``cleanup=False``, temporary directories containing
         Gaussian-filtered images are preserved.

    :param nii_folder:
        Path to the folder containing the input CTA ``.nii.gz`` files.

    :param mask_folder:
        Path to the folder containing the corresponding CTA masks. Mask files
        must have the same filenames as the CTA volumes.

    :param template_path:
        Path to the reference template image (CTA volume).

    :param template_mask_path:
        Path to the template mask, used as the fixed-image mask.

    :param output_path:
        Base directory for all outputs. Created if missing.
        
        - If ``saving_mode="case"``: creates ``output_path/<PREFIX>/`` for each case
        - If ``saving_mode="folder"``: creates ``output_path/registered/`` and 
          ``output_path/transforms/`` subdirectories

    :param saving_mode:
        Defines how outputs are organized:

        - ``"case"`` — one subfolder per input file with both registered image 
          and transformation (recommended for datasets).  
        - ``"folder"`` — all registered images saved into ``registered/`` subfolder 
          and all transformations saved into ``transforms/`` subfolder. Temporary
          directories are kept if ``cleanup=False``.

    :param cleanup:
        If ``True``, deletes intermediate Gaussian-filtered CTA files.
        In folder mode, also deletes temporary directories.

    :param debug:
        If ``True``, prints detailed information about the registration process
        for each file.

    :param number_histogram_bins:
        Number of histogram bins for Mattes Mutual Information metric.
        Default: 128. Common values: 10, 50, 64, 128.

    :param learning_rate:
        Learning rate for Gradient Descent optimizer.
        Default: 0.0001. Common values: 0.0001-1.0.

    :param number_iterations:
        Maximum number of optimization iterations.
        Default: 2000. Common values: 500-5000.

    :param initialization_strategy:
        Strategy for initializing the transformation. Options:
        
        - ``sitk.CenteredTransformInitializerFilter.MOMENTS`` (default) — 
          align based on image moments (center of mass)
        - ``sitk.CenteredTransformInitializerFilter.GEOMETRY`` — 
          align based on image geometry (center and orientation)

    :param sigma_first:
        Standard deviation for the first Gaussian smoothing filter.
        Default: 2.0.

    :param sigma_second:
        Standard deviation for the second Gaussian smoothing filter.
        Default: 3.0.

    :param metric_sampling_percentage:
        Percentage of voxels to sample for metric evaluation (0.0-1.0).
        Default: 0.5 (50%).

    :param initial_transform:
        Initial transformation object. If ``None``, defaults to 
        ``sitk.Euler3DTransform()``. Can be any SimpleITK transform type
        (e.g., ``sitk.Euler3DTransform()``, ``sitk.AffineTransform(3)``).

    :raises FileNotFoundError:
        If ``nii_folder`` does not exist or contains no ``.nii.gz`` files.

    :raises ValueError:
        If ``saving_mode`` is not ``"case"`` or ``"folder"``.

    Example
    -------
    >>> from nidataset.preprocessing import register_CTA_dataset
    >>> import SimpleITK as sitk
    >>>
    >>> # basic usage with default parameters - saving mode "case"
    >>> register_CTA_dataset(
    ...     nii_folder="dataset/CTA_raw/",
    ...     mask_folder="dataset/CTA_masks/",
    ...     template_path="templates/CTA_template.nii.gz",
    ...     template_mask_path="templates/CTA_template_mask.nii.gz",
    ...     output_path="output/CTA_processed/",
    ...     saving_mode="case",
    ...     cleanup=True
    ... )
    >>>
    >>> # custom parameters - saving mode "folder"
    >>> register_CTA_dataset(
    ...     nii_folder="dataset/CTA_raw/",
    ...     mask_folder="dataset/CTA_masks/",
    ...     template_path="templates/CTA_template.nii.gz",
    ...     template_mask_path="templates/CTA_template_mask.nii.gz",
    ...     output_path="output/CTA_processed/",
    ...     saving_mode="folder",
    ...     number_histogram_bins=64,
    ...     learning_rate=0.01,
    ...     number_iterations=1000,
    ...     initialization_strategy=sitk.CenteredTransformInitializerFilter.GEOMETRY,
    ...     sigma_first=1.5,
    ...     sigma_second=2.5,
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
    
    # create output directories based on saving mode
    os.makedirs(output_path, exist_ok=True)
    
    if saving_mode == "folder":
        registered_dir = os.path.join(output_path, "registered")
        transforms_dir = os.path.join(output_path, "transforms")
        os.makedirs(registered_dir, exist_ok=True)
        os.makedirs(transforms_dir, exist_ok=True)
    
    # iterate over nii.gz files with tqdm progress bar
    for nii_file in tqdm(nii_files, desc="Processing CTA files", unit="file"):
        # extract the filename prefix
        prefix = os.path.basename(nii_file).replace(".nii.gz", "")

        # paths for input files
        nii_path = os.path.join(nii_folder, nii_file)
        mask_path = os.path.join(mask_folder, f"{prefix}_mask.nii.gz")
        
        # determine the appropriate output folder based on saving mode
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix)
            os.makedirs(case_output_dir, exist_ok=True)

            register_CTA(nii_path, mask_path, template_path, template_mask_path,
                         case_output_dir, cleanup, debug,
                         number_histogram_bins, learning_rate, number_iterations,
                         initialization_strategy, sigma_first, sigma_second,
                         metric_sampling_percentage, initial_transform)
        else:  # saving_mode == "folder"
            # for folder mode, register to a temporary location
            temp_output_dir = os.path.join(output_path, f"_temp_{prefix}")
            os.makedirs(temp_output_dir, exist_ok=True)
            
            register_CTA(nii_path, mask_path, template_path, template_mask_path,
                         temp_output_dir, cleanup, debug,
                         number_histogram_bins, learning_rate, number_iterations,
                         initialization_strategy, sigma_first, sigma_second,
                         metric_sampling_percentage, initial_transform)
            
            # move registered image to registered folder
            src_registered = os.path.join(temp_output_dir, f"{prefix}_registered.nii.gz")
            dst_registered = os.path.join(registered_dir, f"{prefix}_registered.nii.gz")
            if os.path.exists(src_registered):
                os.rename(src_registered, dst_registered)
            
            # move transformation to transforms folder
            src_transform = os.path.join(temp_output_dir, f"{prefix}_transformation.tfm")
            dst_transform = os.path.join(transforms_dir, f"{prefix}_transformation.tfm")
            if os.path.exists(src_transform):
                os.rename(src_transform, dst_transform)
            
            # if cleanup is True, remove the temporary directory
            if cleanup and os.path.exists(temp_output_dir):
                # remove all files in temp directory
                for file in os.listdir(temp_output_dir):
                    os.remove(os.path.join(temp_output_dir, file))
                os.rmdir(temp_output_dir)
    
    if debug:
        print(f"\nRegistration completed for all files in '{nii_folder}'.")


def register_mask(mask_path: str,
                  transform_path: str,
                  registered_path: str,
                  output_path: str,
                  is_binary: bool = True,
                  debug: bool = False) -> None:
    """
    Apply a saved spatial transformation to a mask using a registered image
    as reference.

    This function propagates a mask through a previously computed registration
    by resampling it into the space of a registered image. It is typically used
    to align brain masks, vessel masks, segmentation masks, or ROIs after image
    registration workflows.

    .. note::
       - The transformation must already exist (e.g. produced by ``register_CTA``)
       - The registered image defines the target space, resolution, and grid
       - Binary masks are resampled with nearest-neighbor interpolation to
         preserve labels
       - Continuous-valued masks are resampled with linear interpolation
       - The input mask filename is expected to end with ``_mask.nii.gz``

    :param mask_path:
        Path to the input mask file (``*_mask.nii.gz``) to be transformed.

    :param transform_path:
        Path to the saved transformation file (``.tfm``) obtained from a previous
        registration.

    :param registered_path:
        Path to the registered image (``.nii.gz``) that defines the target space.
        This must correspond to the image used when computing the transformation.

    :param output_path:
        Path (including filename) where the transformed mask will be saved.

    :param is_binary:
        If ``True``, uses nearest-neighbor interpolation to preserve discrete
        labels (recommended for binary or multi-class masks).
        If ``False``, uses linear interpolation for continuous-valued masks.
        Default: ``True``.

    :param debug:
        If ``True``, prints additional information about the transformation and
        output path.

    :raises FileNotFoundError:
        If the mask, transformation, or reference image file does not exist.

    :raises ValueError:
        If the input mask filename does not end with ``_mask.nii.gz``.

    Example
    -------
    >>> from nidataset.preprocessing import register_CTA, register_mask
    >>>
    >>> # Step 1: register the CTA image
    >>> register_CTA(
    ...     nii_path="scan.nii.gz",
    ...     mask_path="scan_mask.nii.gz",
    ...     template_path="template.nii.gz",
    ...     template_mask_path="template_mask.nii.gz",
    ...     output_path="registered/",
    ...     debug=True
    ... )
    >>>
    >>> # Step 2: apply the same transformation to another mask
    >>> register_mask(
    ...     mask_path="scan_vessel_mask.nii.gz",
    ...     transform_path="registered/scan_transformation.tfm",
    ...     registered_path="registered/scan_registered.nii.gz",
    ...     output_path="registered/scan_vessel_registered_mask.nii.gz",
    ...     is_binary=True,
    ...     debug=True
    ... )
    """

    # check if input files exist
    for file_path in [mask_path, transform_path, registered_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: the input file '{file_path}' does not exist.")
    
    # ensure the mask is a .nii.gz file
    if not mask_path.endswith("_mask.nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '_mask.nii.gz' file. Got '{mask_path}' instead.")
    
    # create output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # load the input mask
    mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)
    
    # load the reference image (defines target space)
    reference_image = sitk.ReadImage(registered_path, sitk.sitkFloat32)
    
    # load the saved transformation
    transformation = sitk.ReadTransform(transform_path)
    
    # choose interpolation method based on mask type
    if is_binary:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear
    
    # apply the transformation
    registered_mask = sitk.Resample(
        mask,
        reference_image,
        transformation,
        interpolator,
        0.0,
        mask.GetPixelID()
    )
    
    # save the transformed mask
    sitk.WriteImage(registered_mask, output_path)
    
    if debug:
        print(f"\nRegistered mask saved at: '{output_path}'.")


def register_mask_dataset(mask_folder: str,
                          transform_folder: str,
                          registered_folder: str,
                          output_path: str,
                          is_binary: bool = True,
                          saving_mode: str = "case",
                          debug: bool = False) -> None:
    """
    Apply saved spatial transformations to all mask files in a dataset folder.

    This function iterates over a folder of mask volumes and propagates each mask
    through its corresponding registration by resampling it into the space of a
    previously registered reference image. It is designed for dataset-level
    workflows where registrations have already been computed.

    .. note::
       - Transformation files must already exist (e.g. generated by
         ``register_CTA_dataset``)
       - Mask filenames are matched to transformations and reference images by
         filename prefix
       - Binary masks are resampled using nearest-neighbor interpolation to
         preserve discrete labels
       - Continuous-valued masks are resampled using linear interpolation
       - The folder structure and file lookup depend on ``saving_mode``

       Saving modes:
       - ``"case"``: transformations and registered images are expected inside
         per-case subfolders ``<PREFIX>/``
       - ``"folder"``: transformations and registered images are searched
         directly in the specified folders by prefix matching

    :param mask_folder:
        Path to the directory containing input mask files (``.nii.gz``).

    :param transform_folder:
        Path to the directory containing transformation files.

        - If ``saving_mode="case"``:
          ``transform_folder/<PREFIX>/<PREFIX>_transformation.tfm``
        - If ``saving_mode="folder"``:
          ``transform_folder/<PREFIX>_transformation.tfm``

    :param registered_folder:
        Path to the directory containing registered reference images.

        - If ``saving_mode="case"``:
          ``registered_folder/<PREFIX>/<PREFIX>_registered.nii.gz``
        - If ``saving_mode="folder"``:
          ``registered_folder/<PREFIX>_registered.nii.gz``

    :param output_path:
        Base output directory where transformed masks are saved. Created if it
        does not exist.

        - If ``saving_mode="case"``: creates ``output_path/<PREFIX>/`` for each case
        - If ``saving_mode="folder"``: saves all outputs directly in
          ``output_path/``

    :param is_binary:
        If ``True``, uses nearest-neighbor interpolation to preserve discrete
        labels (recommended for binary or multi-class masks).
        If ``False``, uses linear interpolation for continuous-valued masks.
        Default: ``True``.

    :param saving_mode:
        Controls how transformations are searched and how outputs are organized.

        - ``"case"`` — per-case subdirectory structure (recommended for datasets)
        - ``"folder"`` — flat folder structure with prefix-based file matching

    :param debug:
        If ``True``, prints detailed information and warnings for each processed
        mask.

    :raises FileNotFoundError:
        If ``mask_folder`` does not exist or contains no ``.nii.gz`` files.

    :raises ValueError:
        If ``saving_mode`` is not ``"case"`` or ``"folder"``.

    Example
    -------
    >>> from nidataset.preprocessing import register_CTA_dataset, register_mask_dataset
    >>>
    >>> # Step 1: register all CTA images (case mode)
    >>> register_CTA_dataset(
    ...     nii_folder="scans/",
    ...     mask_folder="brain_masks/",
    ...     template_path="template.nii.gz",
    ...     template_mask_path="template_mask.nii.gz",
    ...     output_path="registered/",
    ...     saving_mode="case",
    ...     cleanup=True
    ... )
    >>>
    >>> # Step 2: propagate vessel masks using the same transformations
    >>> register_mask_dataset(
    ...     mask_folder="vessel_masks/",
    ...     transform_folder="registered/",
    ...     registered_folder="registered/",
    ...     output_path="registered_vessel_masks/",
    ...     is_binary=True,
    ...     saving_mode="case",
    ...     debug=True
    ... )
    >>>
    >>> # Alternative: flat folder-based workflow
    >>> register_mask_dataset(
    ...     mask_folder="vessel_masks/",
    ...     transform_folder="registered/transforms/",
    ...     registered_folder="registered/registered/",
    ...     output_path="registered_vessel_masks/",
    ...     saving_mode="folder"
    ... )
    """

    # check if mask folder exists
    if not os.path.isdir(mask_folder):
        raise FileNotFoundError(f"Error: the mask folder '{mask_folder}' does not exist.")
    
    # get all .nii.gz files in the mask folder
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".nii.gz")]
    
    # check if there are mask files
    if not mask_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{mask_folder}'.")
    
    # validate saving_mode
    if saving_mode not in ["case", "folder"]:
        raise ValueError("Error: saving_mode must be either 'case' or 'folder'.")
    
    # create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # iterate over mask files with progress bar
    for mask_file in tqdm(mask_files, desc="Processing masks", unit="mask"):
        # extract the filename prefix
        prefix = os.path.basename(mask_file).replace("_mask.nii.gz", "")
        
        # construct mask path
        mask_path = os.path.join(mask_folder, mask_file)
        
        # find transformation and reference files based on saving mode
        if saving_mode == "case":
            # search in case subdirectories: transform_folder/<PREFIX>/
            transform_file = os.path.join(transform_folder, prefix, f"{prefix}_transformation.tfm")
            registered_file = os.path.join(registered_folder, prefix, f"{prefix}_registered.nii.gz")
            
        else:  # saving_mode == "folder"
            # search directly in folders by prefix matching
            transform_file = os.path.join(transform_folder, f"{prefix}_transformation.tfm")
            registered_file = os.path.join(registered_folder, f"{prefix}_registered.nii.gz")
        
        # check if files exist
        if not os.path.exists(transform_file):
            if debug:
                print(f"\nWarning: Transformation file not found at '{transform_file}', skipping...")
            continue
        
        if not os.path.exists(registered_file):
            if debug:
                print(f"\nWarning: Reference image not found at '{registered_file}', skipping...")
            continue
        
        # determine output path based on saving mode
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix)
            os.makedirs(case_output_dir, exist_ok=True)
            output_file = os.path.join(case_output_dir, f"{prefix}_registered_mask.nii.gz")
        else:
            output_file = os.path.join(output_path, f"{prefix}_registered_mask.nii.gz")
        
        # register the mask
        register_mask(
            mask_path=mask_path,
            transform_path=transform_file,
            registered_path=registered_file,
            output_path=output_file,
            is_binary=is_binary,
            debug=debug
        )
    
    if debug:
        print(f"\nMask registration completed for all files in '{mask_folder}'.")


def register_annotation(annotation_path: str,
                       transform_path: str,
                       registered_path: str,
                       output_path: str,
                       recalculate_bbox: bool = True,
                       debug: bool = False) -> None:
    """
    Apply a saved spatial transformation to an annotation and optionally
    recalculate a tight axis-aligned bounding box.

    This function propagates an annotation volume (typically a bounding-box mask)
    through a previously computed registration by resampling it into the space
    of a registered reference image. After transformation, it can either preserve
    the exact deformed annotation or recompute a new tight, axis-aligned bounding
    box enclosing the transformed region.

    .. note::
       - The transformation must already exist (e.g. generated by ``register_CTA``)
       - The registered image defines the target space, resolution, and grid
       - Nearest-neighbor interpolation is used to preserve discrete labels
       - ``recalculate_bbox=True`` is recommended when annotations represent
         axis-aligned bounding boxes and the registration includes rotations
       - If the transformed annotation contains no foreground voxels, an empty
         mask is saved

    :param annotation_path:
        Path to the input annotation file (``*_bbox.nii.gz``), typically encoding
        a bounding box or binary region of interest.

    :param transform_path:
        Path to the saved transformation file (``.tfm``) obtained from a previous
        registration.

    :param registered_path:
        Path to the registered reference image (``.nii.gz``) that defines the
        target space and sampling grid.

    :param output_path:
        Path (including filename) where the transformed annotation will be saved.

    :param recalculate_bbox:
        If ``True``, computes a new tight, axis-aligned bounding box around the
        transformed region.
        If ``False``, preserves the exact deformed annotation shape after
        transformation.
        Default: ``True``.

    :param debug:
        If ``True``, prints detailed information about the transformation,
        bounding box recalculation, and output path.

    :raises FileNotFoundError:
        If the annotation, transformation, or reference image file does not exist.

    :raises ValueError:
        If the input annotation filename does not end with ``_bbox.nii.gz``.

    Example
    -------
    >>> from nidataset.preprocessing import register_CTA, register_annotation
    >>>
    >>> # Step 1: register the CTA image
    >>> register_CTA(
    ...     nii_path="scan.nii.gz",
    ...     mask_path="scan_mask.nii.gz",
    ...     template_path="template.nii.gz",
    ...     template_mask_path="template_mask.nii.gz",
    ...     output_path="registered/",
    ...     debug=True
    ... )
    >>>
    >>> # Step 2: transform annotation with bounding box recalculation
    >>> register_annotation(
    ...     annotation_path="scan_lesion_bbox.nii.gz",
    ...     transform_path="registered/scan_transformation.tfm",
    ...     registered_path="registered/scan_registered.nii.gz",
    ...     output_path="registered/scan_lesion_bbox_registered_mask.nii.gz",
    ...     recalculate_bbox=True,
    ...     debug=True
    ... )
    >>>
    >>> # Alternative: preserve the exact deformed annotation
    >>> register_annotation(
    ...     annotation_path="scan_lesion_precise_bbox.nii.gz",
    ...     transform_path="registered/scan_transformation.tfm",
    ...     registered_path="registered/scan_registered.nii.gz",
    ...     output_path="registered/scan_lesion_precise_registered.nii.gz",
    ...     recalculate_bbox=False,
    ...     debug=True
    ... )
    """

    # check if input files exist
    for file_path in [annotation_path, transform_path, registered_path]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: the input file '{file_path}' does not exist.")
    
    # ensure the annotation is a .nii.gz file
    if not annotation_path.endswith("_bbox.nii.gz"):
        raise ValueError(f"Error: invalid file format. Expected a '_bbox.nii.gz' file. Got '{annotation_path}' instead.")
    
    # create output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # load the annotation
    annotation = sitk.ReadImage(annotation_path, sitk.sitkFloat32)
    
    # load the reference image (defines target space)
    reference_image = sitk.ReadImage(registered_path, sitk.sitkFloat32)
    
    # load the saved transformation
    transformation = sitk.ReadTransform(transform_path)
    
    # apply the transformation to the annotation
    transformed_annotation = sitk.Resample(
        annotation,
        reference_image,
        transformation,
        sitk.sitkNearestNeighbor,
        0.0,
        annotation.GetPixelID()
    )
    
    if recalculate_bbox:
        # convert to numpy array for bbox calculation
        transformed_array = sitk.GetArrayFromImage(transformed_annotation)
        
        # find nonzero voxels
        nonzero_coords = np.argwhere(transformed_array > 0)
        
        # if annotation is empty after transformation, save empty mask
        if nonzero_coords.size == 0:
            sitk.WriteImage(transformed_annotation, output_path)
            if debug:
                print(f"\nWarning: Transformed annotation is empty.")
                print(f"Registered annotation saved at: '{output_path}'.")
            return
        
        # find min and max indices for each axis
        z_min, y_min, x_min = nonzero_coords.min(axis=0)
        z_max, y_max, x_max = nonzero_coords.max(axis=0)
        
        # create new mask with tight bounding box
        new_annotation_array = np.zeros_like(transformed_array, dtype=np.uint8)
        new_annotation_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = 1
        
        # convert back to SimpleITK
        final_annotation = sitk.GetImageFromArray(new_annotation_array)
        final_annotation.CopyInformation(reference_image)
        
        # save the new bounding box
        sitk.WriteImage(final_annotation, output_path)
        
        if debug:
            print(f"\nBounding box recalculated:")
            print(f"  Original bbox size: {nonzero_coords.shape[0]} voxels")
            print(f"  New bbox: [{x_min}:{x_max}, {y_min}:{y_max}, {z_min}:{z_max}]")
            print(f"Registered annotation saved at: '{output_path}'.")
    else:
        # save the deformed annotation without recalculation
        sitk.WriteImage(transformed_annotation, output_path)
        
        if debug:
            print(f"\nRegistered annotation saved at: '{output_path}'.")


def register_annotation_dataset(annotation_folder: str,
                                transform_folder: str,
                                registered_folder: str,
                                output_path: str,
                                recalculate_bbox: bool = True,
                                saving_mode: str = "case",
                                debug: bool = False) -> None:
    """
    Apply saved spatial transformations to all annotation files in a dataset folder.

    This function iterates over a folder of annotation volumes (typically bounding
    box masks) and propagates each annotation through its corresponding registration
    by resampling it into the space of a previously registered reference image.
    Optionally, new tight axis-aligned bounding boxes can be recomputed after
    transformation.

    .. note::
       - Transformation files must already exist (e.g. generated by
         ``register_CTA_dataset``)
       - Annotation filenames are matched to transformations and reference images
         by filename prefix
       - Nearest-neighbor interpolation is used to preserve discrete labels
       - ``recalculate_bbox=True`` is recommended when annotations represent
         axis-aligned bounding boxes and the registration includes rotations
       - The folder structure and file lookup depend on ``saving_mode``

       Saving modes:
       - ``"case"``: transformations and registered images are expected inside
         per-case subfolders ``<PREFIX>/``
       - ``"folder"``: transformations and registered images are searched
         directly in the specified folders by prefix matching

    :param annotation_folder:
        Path to the directory containing input annotation files (``.nii.gz``),
        typically encoding bounding boxes or regions of interest.

    :param transform_folder:
        Path to the directory containing transformation files.

        - If ``saving_mode="case"``:
          ``transform_folder/<PREFIX>/<PREFIX>_transformation.tfm``
        - If ``saving_mode="folder"``:
          ``transform_folder/<PREFIX>_transformation.tfm``

    :param registered_folder:
        Path to the directory containing registered reference images.

        - If ``saving_mode="case"``:
          ``registered_folder/<PREFIX>/<PREFIX>_registered.nii.gz``
        - If ``saving_mode="folder"``:
          ``registered_folder/<PREFIX>_registered.nii.gz``

    :param output_path:
        Base output directory where transformed annotations are saved. Created
        if it does not exist.

        - If ``saving_mode="case"``: creates ``output_path/<PREFIX>/`` for each case
        - If ``saving_mode="folder"``: saves all outputs directly in
          ``output_path/``

    :param recalculate_bbox:
        If ``True``, computes new tight, axis-aligned bounding boxes around the
        transformed regions.
        If ``False``, preserves the exact deformed annotation shapes after
        transformation.
        Default: ``True``.

    :param saving_mode:
        Controls how transformations are searched and how outputs are organized.

        - ``"case"`` — per-case subdirectory structure (recommended for datasets)
        - ``"folder"`` — flat folder structure with prefix-based file matching

    :param debug:
        If ``True``, prints detailed information and warnings for each processed
        annotation.

    :raises FileNotFoundError:
        If ``annotation_folder`` does not exist or contains no ``.nii.gz`` files.

    :raises ValueError:
        If ``saving_mode`` is not ``"case"`` or ``"folder"``.

    Example
    -------
    >>> from nidataset.preprocessing import register_CTA_dataset, register_annotation_dataset
    >>>
    >>> # Step 1: register all CTA images (case mode)
    >>> register_CTA_dataset(
    ...     nii_folder="scans/",
    ...     mask_folder="brain_masks/",
    ...     template_path="template.nii.gz",
    ...     template_mask_path="template_mask.nii.gz",
    ...     output_path="registered/",
    ...     saving_mode="case",
    ...     cleanup=True
    ... )
    >>>
    >>> # Step 2: propagate lesion annotations using the same transformations
    >>> register_annotation_dataset(
    ...     annotation_folder="lesion_bboxes/",
    ...     transform_folder="registered/",
    ...     registered_folder="registered/",
    ...     output_path="registered_lesions/",
    ...     recalculate_bbox=True,
    ...     saving_mode="case",
    ...     debug=True
    ... )
    >>>
    >>> # Alternative: flat folder-based workflow
    >>> register_annotation_dataset(
    ...     annotation_folder="lesion_bboxes/",
    ...     transform_folder="registered/transforms/",
    ...     registered_folder="registered/registered/",
    ...     output_path="registered_lesions/",
    ...     saving_mode="folder"
    ... )
    """

    # check if annotation folder exists
    if not os.path.isdir(annotation_folder):
        raise FileNotFoundError(f"Error: the annotation folder '{annotation_folder}' does not exist.")
    
    # get all .nii.gz files in the annotation folder
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".nii.gz")]
    
    # check if there are annotation files
    if not annotation_files:
        raise FileNotFoundError(f"Error: no .nii.gz files found in '{annotation_folder}'.")
    
    # validate saving_mode
    if saving_mode not in ["case", "folder"]:
        raise ValueError("Error: saving_mode must be either 'case' or 'folder'.")
    
    # create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # iterate over annotation files with progress bar
    for annotation_file in tqdm(annotation_files, desc="Processing annotations", unit="annotation"):
        # extract the filename prefix
        prefix = os.path.basename(annotation_file).replace("_bbox.nii.gz", "")
        
        # construct annotation path
        annotation_path = os.path.join(annotation_folder, annotation_file)
        
        # find transformation and reference files based on saving mode
        if saving_mode == "case":
            # search in case subdirectories: transform_folder/<PREFIX>/
            transform_file = os.path.join(transform_folder, prefix, f"{prefix}_transformation.tfm")
            registered_file = os.path.join(registered_folder, prefix, f"{prefix}_registered.nii.gz")
            
        else:  # saving_mode == "folder"
            # search directly in folders by prefix matching
            transform_file = os.path.join(transform_folder, f"{prefix}_transformation.tfm")
            registered_file = os.path.join(registered_folder, f"{prefix}_registered.nii.gz")
        
        # check if files exist
        if not os.path.exists(transform_file):
            if debug:
                print(f"\nWarning: Transformation file not found at '{transform_file}', skipping...")
            continue
        
        if not os.path.exists(registered_file):
            if debug:
                print(f"\nWarning: Reference image not found at '{registered_file}', skipping...")
            continue
        
        # determine output path based on saving mode
        if saving_mode == "case":
            case_output_dir = os.path.join(output_path, prefix)
            os.makedirs(case_output_dir, exist_ok=True)
            output_file = os.path.join(case_output_dir, f"{prefix}_registered_bbox.nii.gz")
        else:
            output_file = os.path.join(output_path, f"{prefix}_registered_bbox.nii.gz")
        
        # register the annotation
        register_annotation(
            annotation_path=annotation_path,
            transform_path=transform_file,
            registered_path=registered_file,
            output_path=output_file,
            recalculate_bbox=recalculate_bbox,
            debug=debug
        )
    
    if debug:
        print(f"\nAnnotation registration completed for all files in '{annotation_folder}'.")

        
