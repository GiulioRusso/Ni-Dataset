"""
``nii`` — command-line entry point for quick NIfTI operations.

Thin wrapper over the Python API: rotate, flip, crop, crop-to-content, rescale
intensities, and convert to/from DICOM, straight from the shell. For the
NIfTI-input commands, PATH may be a single file or a folder (processed as a
dataset). Every command prints the output path(s) it wrote.

Exit codes: ``0`` success, ``2`` usage / runtime error.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


def _run_on_path(path: str, output_path: str, single, dataset, **kwargs) -> List[str]:
    """Dispatch to the single-file or dataset function based on PATH."""
    if os.path.isdir(path):
        return dataset(path, output_path, **kwargs)
    return [single(path, output_path, **kwargs)]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nii",
        description="Quick NIfTI operations (rotate, flip, crop, rescale, DICOM conversion).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  nii rotate scan.nii.gz out/ --k 1 --axes 0 1\n"
            "  nii flip scans/ out/ --axis 0\n"
            "  nii crop scan.nii.gz out/ --bbox 10 100 10 100 5 60\n"
            "  nii crop-content scan.nii.gz out/ --margin 2\n"
            "  nii rescale scan.nii.gz out/ --out-min 0 --out-max 255\n"
            "  nii to-dicom scan.nii.gz out/\n"
            "  nii from-dicom dicom/case_01/ out/\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_io(sp, path_help="NIfTI file or folder."):
        sp.add_argument("path", help=path_help)
        sp.add_argument("output_path", help="Output directory.")

    sp = sub.add_parser("rotate", help="Rotate by k*90 degrees in a voxel plane.")
    add_io(sp)
    sp.add_argument("--k", type=int, default=1, help="Number of 90-degree rotations (default 1).")
    sp.add_argument("--axes", type=int, nargs=2, default=[0, 1], metavar=("A", "B"),
                    help="Rotation-plane axes (default 0 1).")

    sp = sub.add_parser("flip", help="Mirror along a voxel axis.")
    add_io(sp)
    sp.add_argument("--axis", type=int, default=0, help="Voxel axis to flip (default 0).")

    sp = sub.add_parser("crop", help="Crop to an explicit voxel box (single file).")
    add_io(sp, path_help="NIfTI file.")
    sp.add_argument("--bbox", type=int, nargs=6, required=True,
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="Half-open voxel bounds x0 x1 y0 y1 z0 z1.")

    sp = sub.add_parser("crop-content", help="Crop to the minimal foreground box.")
    add_io(sp)
    sp.add_argument("--threshold", type=float, default=None,
                    help="Foreground cutoff (default: keep non-zero).")
    sp.add_argument("--margin", type=int, default=0, help="Voxels of padding to keep (default 0).")

    sp = sub.add_parser("rescale", help="Linearly rescale intensities to a range.")
    add_io(sp)
    sp.add_argument("--out-min", type=float, default=0.0, help="Output min (default 0).")
    sp.add_argument("--out-max", type=float, default=1.0, help="Output max (default 1).")
    sp.add_argument("--in-min", type=float, default=None, help="Input min (default: data min).")
    sp.add_argument("--in-max", type=float, default=None, help="Input max (default: data max).")

    sp = sub.add_parser("to-dicom", help="Convert NIfTI to a DICOM series.")
    add_io(sp)
    sp.add_argument("--series-description", default="nidataset", help="DICOM SeriesDescription.")

    sp = sub.add_parser("from-dicom", help="Convert a DICOM series folder to NIfTI.")
    sp.add_argument("path", help="Directory of DICOM slices (one series).")
    sp.add_argument("output_path", help="Output directory.")

    sp = sub.add_parser("preview", help="Render a slice-montage PNG of a volume.")
    add_io(sp, path_help="NIfTI file.")
    sp.add_argument("--view", default="axial", choices=("axial", "coronal", "sagittal"),
                    help="Anatomical view (default axial).")
    sp.add_argument("--num-slices", type=int, default=16, help="Slices in the montage (default 16).")
    sp.add_argument("--cols", type=int, default=4, help="Grid columns (default 4).")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Run the ``nii`` CLI. Returns the process exit code."""
    # Imported here so `nii --help` stays fast and import errors surface per-command.
    from .spatial import (
        rotate_volume, rotate_volume_dataset,
        flip_volume, flip_volume_dataset,
        crop_volume,
        crop_to_content, crop_to_content_dataset,
    )
    from .transforms import (
        rescale_intensity, rescale_intensity_dataset,
        dicom_to_nifti, nifti_to_dicom,
    )
    from .visualization import create_slice_montage

    args = _build_parser().parse_args(argv)

    try:
        if args.command == "rotate":
            outs = _run_on_path(args.path, args.output_path, rotate_volume, rotate_volume_dataset,
                                k=args.k, axes=tuple(args.axes))
        elif args.command == "flip":
            outs = _run_on_path(args.path, args.output_path, flip_volume, flip_volume_dataset,
                                axis=args.axis)
        elif args.command == "crop":
            if os.path.isdir(args.path):
                raise ValueError("crop takes a single NIfTI file, not a folder.")
            outs = [crop_volume(args.path, args.output_path, bbox=tuple(args.bbox))]
        elif args.command == "crop-content":
            outs = _run_on_path(args.path, args.output_path,
                                crop_to_content, crop_to_content_dataset,
                                threshold=args.threshold, margin=args.margin)
        elif args.command == "rescale":
            outs = _run_on_path(args.path, args.output_path,
                                rescale_intensity, rescale_intensity_dataset,
                                out_min=args.out_min, out_max=args.out_max,
                                in_min=args.in_min, in_max=args.in_max)
        elif args.command == "to-dicom":
            if os.path.isdir(args.path):
                from ._helpers import list_nifti_files
                outs = [nifti_to_dicom(os.path.join(args.path, f), args.output_path,
                                       series_description=args.series_description)
                        for f in list_nifti_files(args.path)]
            else:
                outs = [nifti_to_dicom(args.path, args.output_path,
                                       series_description=args.series_description)]
        elif args.command == "from-dicom":
            outs = [dicom_to_nifti(args.path, args.output_path)]
        elif args.command == "preview":
            if os.path.isdir(args.path):
                raise ValueError("preview takes a single NIfTI file, not a folder.")
            outs = [create_slice_montage(args.path, args.output_path, view=args.view,
                                         num_slices=args.num_slices, cols=args.cols)]
        else:  # pragma: no cover - argparse enforces a valid command
            raise ValueError(f"unknown command '{args.command}'")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for out in outs:
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
