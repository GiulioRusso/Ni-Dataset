# nidataset/__init__.py

from ._version import __version__

from .draw import (draw_3D_boxes,
                   draw_2D_annotations,
                   from_2D_to_3D_coords)

from .preprocessing import (skull_CTA,
                            skull_CTA_dataset,
                            mip,
                            mip_dataset,
                            resampling,
                            resampling_dataset,
                            register_CTA,
                            register_CTA_dataset,
                            register_mask,
                            register_mask_dataset,
                            register_annotation,
                            register_annotation_dataset)

from .slices import (extract_slices,
                     extract_slices_dataset,
                     extract_annotations,
                     extract_annotations_dataset)

from .volume import (swap_nifti_views,
                     extract_bounding_boxes,
                     extract_bounding_boxes_dataset,
                     generate_brain_mask,
                     generate_brain_mask_dataset,
                     crop_and_pad,
                     crop_and_pad_dataset,
                     generate_heatmap_volume)

from .utility import (dataset_images_info,
                      dataset_annotations_info)

from .analysis import (compare_volumes,
                       compare_volumes_dataset,
                       compute_volume_statistics,
                       compute_volume_statistics_dataset,
                       split_dataset)

from .transforms import (intensity_normalization,
                         intensity_normalization_dataset,
                         windowing,
                         windowing_dataset,
                         resample_to_reference,
                         resample_to_reference_dataset,
                         apply_transform,
                         nifti_to_numpy,
                         numpy_to_nifti,
                         CT_WINDOW_PRESETS)

from .visualization import (overlay_mask_on_volume,
                            overlay_mask_on_volume_dataset,
                            create_slice_montage)

__all__ = [
    "__version__",
    # draw
    "draw_3D_boxes",
    "draw_2D_annotations",
    "from_2D_to_3D_coords",
    # preprocessing
    "skull_CTA",
    "skull_CTA_dataset",
    "mip",
    "mip_dataset",
    "resampling",
    "resampling_dataset",
    "register_CTA",
    "register_CTA_dataset",
    "register_mask",
    "register_mask_dataset",
    "register_annotation",
    "register_annotation_dataset",
    # slices
    "extract_slices",
    "extract_slices_dataset",
    "extract_annotations",
    "extract_annotations_dataset",
    # volume
    "swap_nifti_views",
    "extract_bounding_boxes",
    "extract_bounding_boxes_dataset",
    "generate_brain_mask",
    "generate_brain_mask_dataset",
    "crop_and_pad",
    "crop_and_pad_dataset",
    "generate_heatmap_volume",
    # utility
    "dataset_images_info",
    "dataset_annotations_info",
    # analysis
    "compare_volumes",
    "compare_volumes_dataset",
    "compute_volume_statistics",
    "compute_volume_statistics_dataset",
    "split_dataset",
    # transforms
    "intensity_normalization",
    "intensity_normalization_dataset",
    "windowing",
    "windowing_dataset",
    "resample_to_reference",
    "resample_to_reference_dataset",
    "apply_transform",
    "nifti_to_numpy",
    "numpy_to_nifti",
    "CT_WINDOW_PRESETS",
    # visualization
    "overlay_mask_on_volume",
    "overlay_mask_on_volume_dataset",
    "create_slice_montage",
]
