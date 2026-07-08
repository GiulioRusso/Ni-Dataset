---
title: Package Functions
nav_order: 3
has_children: true
---

# 📦 Package Functions

Core API for NIfTI dataset management, grouped by source module. Each module below
is a subsection with one page per function.

| Module | `nidataset.` | Covers |
|--------|--------------|--------|
| [Analysis](analysis/) | `analysis` | Volume comparison, statistics, dataset splitting. |
| [Draw](draw/) | `draw` | 2D/3D box and annotation drawing, coordinate mapping. |
| [Preprocessing](preprocessing/) | `preprocessing` | Skull stripping, MIP, resampling, registration. |
| [Slices](slices/) | `slices` | 2D slice and annotation extraction. |
| [Transforms](transforms/) | `transforms` | Intensity normalization, windowing, resampling, NumPy conversion. |
| [Utility](utility/) | `utility` | Dataset image / annotation inspection. |
| [Visualization](visualization/) | `visualization` | Mask overlays and slice montages. |
| [Volume](volume/) | `volume` | View swapping, bounding boxes, brain masks, crop/pad, heatmaps. |

Related sections: [Quality Control (`niqc`)](../qc/) validates datasets;
[Quick Operations (`nii`)](../nii/) applies fast geometric / conversion edits from
the shell.
