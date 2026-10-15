"""
Generate a tiny synthetic CT-like dataset for the example pipeline.

Writes, next to this script (or under the folder given as first argument)::

    data/images/case01.nii.gz          head-like ellipsoid + bright lesion
    data/images/case02.nii.gz          same head shifted (2, -2, 1) and rotated 10 deg, lesion elsewhere
    data/masks/caseNN_mask.nii.gz      brain mask (ellipsoid)
    data/annotations/caseNN_bbox.nii.gz  lesion bounding box (label 1)
    data/template/template.nii.gz      unshifted head, registration target
    data/template/template_mask.nii.gz

No real patient data. Volumes are 40x40x32 so the whole pipeline runs in seconds.

Usage::

    python make_synthetic.py [ROOT]
"""

import os
import sys

import nibabel as nib
import numpy as np

SHAPE = (40, 40, 32)
AFFINE = np.diag([1.0, 1.0, 1.5, 1.0])  # 1 x 1 x 1.5 mm, RAS


def _ellipsoid(center, radii, head_center=None, angle=0.0):
    """Voxels inside an axis-aligned ellipsoid, rotated by *angle* degrees about z through *head_center*."""
    grid = np.indices(SHAPE).astype(np.float32)
    if angle:
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        hx, hy = head_center[0], head_center[1]
        x, y = grid[0] - hx, grid[1] - hy
        grid[0], grid[1] = c * x + s * y + hx, -s * x + c * y + hy  # inverse-rotate the sample grid
    dist = sum(((grid[i] - center[i]) / radii[i]) ** 2 for i in range(3))
    return dist <= 1.0


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(data, AFFINE), path)


def make(root):
    rng = np.random.default_rng(0)
    data = os.path.join(root, "data")
    radii = (15, 17, 12)

    # Internal structures (offset from head center, radii, intensity), so the anatomy moves
    # with the head: registration has real features to align, not a flat or ramped blob.
    structures = [((0, -3, 0), (4, 6, 3), 15.0), ((5, 6, 2), (3, 3, 3), 60.0), ((-6, 4, -2), (3, 2, 2), 25.0)]

    def head(center, angle=0.0):
        mask = _ellipsoid(center, radii, center, angle)
        image = np.where(mask, 40.0, 0.0)
        for offset, r, value in structures:
            image[_ellipsoid(np.add(center, offset), r, center, angle)] = value
        return mask, image + rng.normal(0, 2, SHAPE)

    template_mask, template = head((20, 20, 16))
    template = np.clip(template, 0, None).astype(np.float32)
    _save(os.path.join(data, "template", "template.nii.gz"), template)
    _save(os.path.join(data, "template", "template_mask.nii.gz"), template_mask.astype(np.uint8))

    # (head center, rotation about z in degrees, lesion bbox lo..hi inclusive). case01's
    # lesion spans 12..19 on every axis, so slice 015 shows it in all three views (used
    # by the draw step). case02 is shifted and rotated: registration has work to do.
    cases = {
        "case01": ((20, 20, 16), 0.0, (12, 19)),
        "case02": ((22, 18, 17), 10.0, (20, 25)),
    }
    for name, (center, angle, (lo, hi)) in cases.items():
        mask, image = head(center, angle)
        image[lo : hi + 1, lo : hi + 1, lo : hi + 1] = 70.0
        bbox = np.zeros(SHAPE, dtype=np.uint8)
        bbox[lo : hi + 1, lo : hi + 1, lo : hi + 1] = 1
        _save(os.path.join(data, "images", f"{name}.nii.gz"), np.clip(image, 0, None).astype(np.float32))
        _save(os.path.join(data, "masks", f"{name}_mask.nii.gz"), mask.astype(np.uint8))
        _save(os.path.join(data, "annotations", f"{name}_bbox.nii.gz"), bbox)
    return data


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    print(make(root))
