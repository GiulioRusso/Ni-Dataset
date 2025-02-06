import nidataset as nid


def main():
    print("| -------------------------------- |")
    print("| NIfTI Dataset Management Example |")
    print("| -------------------------------- |\n")

    # ------ #
    # SLICES #
    # ------ #
    nifti_file = "./dataset/toy-CTA.nii.gz"
    nid.Slices.extract_slices(nii_path=nifti_file,
                              output_dir="./output/extracted_slices/",
                              view="axial",
                              debug=True)
    
    # ----------- #
    # ANNOTATIONS #
    # ----------- #
    annotation_file = "./dataset/toy-annotation.nii.gz"
    nid.Slices.extract_annotations(nii_path=annotation_file,
                                   output_dir="./output/annotations/",
                                   view="axial",
                                   saving_mode="slice",
                                   data_mode="center",
                                   debug=True)
    
    # ----------------- #
    # VOLUME PROCESSING #
    # ----------------- #
    nid.Volume.swap_nifti_views(nii_path=nifti_file,
                                output_dir="./output/processed_volumes/",
                                source_view="axial",
                                target_view="coronal",
                                debug=True)
    
    print("Processing complete. Results saved in ./output/")


if __name__ == "__main__":
    main()
