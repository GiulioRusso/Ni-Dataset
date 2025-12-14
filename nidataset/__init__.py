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
                     crop_and_pad_dataset)

from .utility import (dataset_images_info,
                      dataset_annotations_info)

__all__ = [
    "__version__",
    "draw_3D_boxes",
    "draw_2D_annotations",
    "from_2D_to_3D_coords",
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
    "extract_slices",
    "extract_slices_dataset",
    "extract_annotations",
    "extract_annotations_dataset",
    "swap_nifti_views",
    "extract_bounding_boxes",
    "extract_bounding_boxes_dataset",
    "generate_brain_mask",
    "generate_brain_mask_dataset",
    "crop_and_pad",
    "crop_and_pad_dataset",
    "dataset_images_info",
    "dataset_annotations_info"
]