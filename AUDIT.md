# Correctness audit — `nidataset`

Branch: `bugfix/audit`. One concern per commit; every fix ships a regression
test that fails on the pre-fix code and passes after. All tests use synthetic
in-memory NIfTI volumes (`tests/`), no real data.

## Module / public-API map

Public surface (`nid.<name>`, from `__init__.py`, 47 functions + `CT_WINDOW_PRESETS`):

| Module | Public functions |
|--------|------------------|
| `draw` | `draw_3D_boxes`, `draw_2D_annotations`, `from_2D_to_3D_coords` |
| `preprocessing` | `skull_CTA(_dataset)`, `mip(_dataset)`, `resampling(_dataset)`, `register_CTA(_dataset)`, `register_mask(_dataset)`, `register_annotation(_dataset)` |
| `slices` | `extract_slices(_dataset)`, `extract_annotations(_dataset)` |
| `volume` | `swap_nifti_views`, `extract_bounding_boxes(_dataset)`, `generate_brain_mask(_dataset)`, `crop_and_pad(_dataset)`, `generate_heatmap_volume` |
| `utility` | `dataset_images_info`, `dataset_annotations_info` |
| `analysis` | `compare_volumes(_dataset)`, `compute_volume_statistics(_dataset)`, `split_dataset` |
| `transforms` | `intensity_normalization(_dataset)`, `windowing(_dataset)`, `resample_to_reference(_dataset)`, `apply_transform`, `nifti_to_numpy`, `numpy_to_nifti`, `CT_WINDOW_PRESETS` |
| `visualization` | `overlay_mask_on_volume(_dataset)`, `create_slice_montage` |

Stack: `nibabel`, `SimpleITK`, `scikit-image`, `scipy`, `opencv-python`, `pandas`,
`Pillow`, `numpy`, `matplotlib`, `tqdm`. Python `>=3.8`, pytest. No new deps added.

Verification: `pip install -e ".[dev]"` clean; full suite green; 38/39 public
functions smoke-tested on synthetic volumes (the 39th was the matplotlib bug
below, now 39/39). `skull_CTA*` (FSL) and `register_*` (Elastix) need external
tools and were audited by reading only — see "Not fixed".

## Findings (sorted by severity)

| # | File:function | Bug class | What was broken | Fix | Changes output? | Test |
|---|---------------|-----------|-----------------|-----|-----------------|------|
| 1 | `volume.py:crop_and_pad` | 1 (affine) | When cropping ran, the affine origin was shifted by the bounding-box minimum only, ignoring the centered-crop offset → cropped volume silently translated in world space (the classic NIfTI crop bug). | Origin shifted by `min_coords + crop_before`; world position of every voxel preserved. | **YES** (corrected geometry) | `tests/test_crop_and_pad.py::test_preserves_world_position` |
| 2 | `volume.py:crop_and_pad` | 4 (indexing) | Centered crop `data[c//2 : -(c//2 or None)]` removed `c-1` voxels on odd over-size and **nothing** when over-size was exactly 1 (`0 or None` → `[0:None]`). Output ≠ requested `target_shape`. | Explicit before/after split, slice `[before : dim-after]`; exact target shape. | **YES** (shape now correct) | `tests/test_crop_and_pad.py::test_odd_crop_yields_exact_target_shape` |
| 3 | `volume.py:swap_nifti_views` | 1, 2 (affine/axes) | Affine permuted matrix **rows** (`affine[:3,:3][new_axes,:]`) instead of columns, never accounted for the `np.rot90` on the data, and left translation untouched → swapped volume mis-placed/mis-oriented in world space. | Derive exact output→input index affine by relabeling a flat-index volume; `new_affine = old @ [[M, o],[0,1]]`. | **YES** (corrected header) | `tests/test_swap_views.py::test_preserves_world_position`, `::test_roundtrip_restores_data_and_affine` |
| 4 | `visualization.py:_get_colormap` | 5/6 (practical) | `matplotlib.cm.get_cmap` was removed in matplotlib 3.9; `pyproject` allows `matplotlib>=3.5`, so a clean install makes `overlay_mask_on_volume[_dataset]` raise `AttributeError` on every call. | Use `matplotlib.colormaps` registry (≥3.6) with `cm.get_cmap` fallback for 3.5. | NO (fixes a crash) | `tests/test_visualization.py::test_overlay_mask_runs_on_modern_matplotlib` |
| 5 | `volume.py:crop_and_pad` | 4 (edge case) | All-zero / no-positive-voxel volume → `np.argwhere` empty → `coords.min()` raised an obscure "zero-size array to reduction" error. | Guard and raise `ValueError` naming the file and cause. | NO (clearer error only) | `tests/test_crop_and_pad.py::test_empty_volume_raises_clear_error` |

## Not fixed (documented, lower risk or out of scope)

- **`slices.py:extract_slices` / `extract_annotations` — fixed axis assumption (class 2).**
  Axial = last axis, etc., is assumed without checking the affine. Correct for
  RAS-ordered `(X,Y,Z)` data (the documented contract) but wrong for volumes
  stored in another voxel order. Fixing means reorienting via the affine, an
  observable behavior change to a core API — flagged, not changed.
- **`draw.py:draw_3D_boxes` (class 6).** Boxes drawn with `data[x_min:x_max]`
  (exclusive max, off-by-one vs the inclusive boxes produced by
  `extract_annotations`); the docstring also promises a `ValueError` on NaN that
  the code never raises. Minor; left to avoid changing drawing convention.
- **`transforms.py:numpy_to_nifti` (class 1).** Will happily write `int64` arrays,
  which some NIfTI readers reject. It's user-supplied data, so not forced here.
- **`transforms.py:intensity_normalization` (cosmetic).** Dead `mapping` dict in
  the histogram branch (the real mapping uses `argsort`). Harmless.
- **`preprocessing.py:register_*`, `skull_CTA*` (external).** Depend on FSL /
  Elastix; not exercised at runtime in this environment. Read statically, no
  obvious geometry defect, but not regression-tested.

## Most likely to change results for existing users (e.g. CT-manager)

These three alter output and warrant a **major version bump**:

1. **#1 `crop_and_pad` affine origin** — cropped volumes were silently shifted in
   world space. Any downstream registration, mask overlay, or world-coordinate
   read off a cropped output was using wrong coordinates; it now matches the
   source volume.
2. **#2 `crop_and_pad` output shape** — outputs were not actually `target_shape`
   on odd crops, breaking fixed-size model-input pipelines. Shapes are now exact.
3. **#3 `swap_nifti_views` affine** — swapped volumes were mis-placed in world
   space. Headers now preserve voxel world positions (voxel data order unchanged).
