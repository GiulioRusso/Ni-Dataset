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
    Performs a skull-stripping pipeline designed for CTA (thresholding, smoothing, FSL BET, and clipping) on a single NIfTI file.

    :implementation note:
        CTAs has to be already centered to the brain area (no robust_fov is applied in this pipeline to ensure the input dimension is kept).
        FSL command line needs to be installed via official site (https://fsl.fmrib.ox.ac.uk/fsldownloads_registration/).
        The file that will use this function needs to be launched from terminal: `python3 main.py`, where `main.py` use this function.

    :param nii_path: Path to the input .nii.gz file with shape (X, Y, Z).
    :param output_path: Directory where intermediate and final outputs will be stored.
    :param f_value: Fractional intensity threshold for BET (skull-stripping).
    :param clip_value: Minimum and maximum intensity values for clipping, e.g. (0, 200).
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
    Performs a skull-stripping pipeline designed for CTA (thresholding, smoothing, FSL BET, and clipping) on the NIfTI files inside the input folder.

    :implementation note:
        CTAs has to be already centered to the brain area (no robust_fov is applied in this pipeline to ensure the input dimension is kept).
        FSL command line needs to be installed via official site (https://fsl.fmrib.ox.ac.uk/fsldownloads_registration/).
        The file that will use this function needs to be launched from terminal: `python3 main.py`, where `main.py` use this function.

    :param nii_folder: Directory containing .nii.gz files to be processed.
    :param output_path: Directory where the skull-stripped .nii.gz files will be saved.
    :param f_value: Fractional intensity threshold for BET (skull-stripping).
    :param clip_value: Minimum and maximum intensity values for clipping, e.g. (0, 200).
    :param cleanup: If True, removes intermediate files (thresholded and skulled images). Mask and clipped images are always retained.
    :param saving_mode: "case" -> creates a dedicated subfolder for each .nii.gz file.
                        "folder" -> saves all the results in a single subfolder.
    :param debug: If True, prints additional information about the process.

    :raises FileNotFoundError: If the dataset folder does not exist or contains no .nii.gz files.
    :raises ValueError: If an invalid saving_mode is provided.
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

        from nidataset.Preprocessing import mip

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

        from nidataset.Preprocessing import mip_dataset

        # define paths
        nii_folder = "path/to/dataset"
        output_path = "path/to/output_directory"

        # choose the anatomical view ('axial', 'coronal', or 'sagittal')
        view = "axial"

        # run the function
        mip_dataset(nii_folder=nii_folder, 
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


def resampling(nii_path: str,
               output_path: str,
               desired_volume: tuple,
               debug: bool = False) -> None:
    """
    Resamples a single NIfTI file to the desired volume size.
    Saves the output file as:

        <Nifti FILENAME>_resampled.nii.gz

    :param nii_path: path to the input .nii.gz file.
    :param output_path: path where the resampled NIfTI file will be saved.
    :param desired_volume: target volume size (X, Y, Z).
    :param debug: if True, prints additional information about the process.
    
    :raises FileNotFoundError: if the input NIfTI file does not exist.
    :raises ValueError: if the file is empty, has invalid dimensions or if the desired volume has invalid dimensions.

    Example:

        from nidataset.Preprocessing import resampling

        # define paths
        nii_path = "path/to/input_image.nii.gz"
        output_path = "path/to/output_directory"

        # run the function
        resampling(nii_path=nii_path,
                   output_path=output_path,
                   desired_volume=(224,224,128),
                   debug=True)
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
    Resamples all NIfTI files in a dataset folder to the desired volume size. Saves the output file as:

        <Nifti FILENAME>_resampled.nii.gz

    :param nii_folder: path to the folder containing .nii.gz files.
    :param output_path: path where the resampled NIfTI files will be saved.
    :param desired_volume: target volume size (X, Y, Z).
    :param saving_mode: "case" -> creates a folder for each case.
                        "folder" -> saves all resampled images inside a single folder.
    :param debug: if True, prints additional information about the process.

    :raises FileNotFoundError: if the dataset folder does not exist or contains no .nii.gz files.
    :raises ValueError: if an invalid saving_mode is provided.

    Example:

        from nidataset.Preprocessing import resampling_dataset

        # define paths
        nii_folder = "path/to/dataset"
        output_path = "path/to/output_directory"

        # run the function
        resampling_dataset(nii_folder=nii_folder, 
                           output_path=output_path, 
                           desired_volume=(224,224,128),
                           saving_mode="case", 
                           debug=True)
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
    Registers a CTA image to a given template using mutual information-based registration. Saves the output registered image and transformation file as:

        <Nifti FILENAME>_registered.nii.gz
        <Nifti FILENAME>_gaussian_filtered.nii.gz
        <Nifti FILENAME>_transformation.tfm

    :param nii_path: Path to the input .nii.gz file.
    :param mask_path: Path to the input mask file.
    :param template_path: Path to the template image file.
    :param template_mask_path: Path to the template mask file.
    :param output_image_path: Path where the registered NIfTI file will be saved.
    :param output_transformation_path: Path where the transformation file will be saved.
    :param cleanup: If True, deletes the temporary gaussian-filtered CTA file.
    :param debug: If True, prints additional information about the process.
    
    :raises FileNotFoundError: If any input file does not exist.
    :raises ValueError: If the file is empty or has invalid dimensions.

    Example:

        from nidataset.Preprocessing import register_CTA

        # define paths
        nii_path = "path/to/input_image.nii.gz"
        output_image_path = "path/to/output_image_directory"
        output_transformation_path = "path/to/output_transformation_directory"

        # run the function
        register_CTA(nii_path=nii_path,
                     mask_path=mask_path,
                     template_path="path/to/template",
                     template_mask_path="path/to/template_mask",
                     output_image_path=output_image_path,
                     output_transformation_path=output_transformation_path,
                     cleanup=True,
                     debug=False)
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
    Registers all CTA images in a dataset folder to a given template using mutual information-based registration.
    Saves the output registered image and transformation file as:

        <Nifti FILENAME>_registered.nii.gz
        <Nifti FILENAME>_gaussian_filtered.nii.gz
        <Nifti FILENAME>_transformation.tfm

    :param nii_folder: Path to the folder containing .nii.gz files.
    :param mask_folder: Path to the folder containing mask files.
    :param template_path: Path to the template image file.
    :param template_mask_path: Path to the template mask file.
    :param output_image_path: Path where the registered NIfTI files will be saved.
    :param output_transformation_path: Path where the transformation files will be saved. Default is empty if saving_mode is "case" and you want to save the transform inside the case folder.
    :param saving_mode: "case" -> creates a folder for each case and save image and transform file.
                        "folder" -> saves all registered images in a single folder.
    :param cleanup: If True, deletes the temporary gaussian-filtered CTA files.
    :param debug: If True, prints additional information about the process.

    :raises FileNotFoundError: If the dataset folder does not exist or contains no .nii.gz files.
    :raises ValueError: If an invalid saving_mode is provided.

    Example:

        from nidataset.Preprocessing import register_CTA_dataset

        # define paths
        nii_folder = "path/to/dataset"
        mask_folder = "path/to/dataset_masks"
        output_image_path = "path/to/output_image_directory"
        output_transformation_path = "path/to/output_transformation_directory"

        # run the function
        register_CTA_dataset(nii_folder=nii_folder,
                             mask_folder=mask_folder,
                             template_path="path/to/template",
                             template_mask_path="path/to/template_mask",
                             output_image_path=output_image_path,
                             output_transformation_path=output_transformation_path,
                             saving_mode="folder",
                             cleanup=True,
                             debug=False)

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

