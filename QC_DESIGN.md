# `nidataset.qc` — Design notes

This document records the choices behind the quality-control module: what counts
as `ok` / `warning` / `error` and **why**, the geometric conventions assumed, and
the default thresholds with their rationale (so you know what you are changing
when you override them via `QCConfig`).

## Goal

Catch the **silent** dataset bugs — the ones that never raise an exception but
quietly corrupt detection/segmentation training: unexpected orientation, a mask
misaligned from its image in world space, empty/out-of-mask annotations,
anisotropic spacing, all-black padding slices, non-portable `int64`, NaN/inf.

A QC tool that is wrong about geometry is **worse than nothing**, because it gives
false confidence. So the geometry checks (affine, orientation, alignment) are the
most carefully implemented part, and each is explicit about its convention.

## Severity model

Every check returns one of three states; a report's status is the worst of its
checks.

| Status      | Meaning                                                                                  |
|-------------|------------------------------------------------------------------------------------------|
| `ok`        | Check passed.                                                                             |
| `warning`   | Suspicious but not provably wrong; may be intentional. **Does not** fail `--strict`.     |
| `error`     | Almost certainly a bug that will corrupt training. Fails `--strict` (CLI exit code `1`). |

Rule of thumb used to assign severity:

- **`error`** = breaks the geometric/semantic contract the data must satisfy:
  missing/singular/non-finite affine, wrong orientation vs an explicit
  expectation, image↔mask shape or affine mismatch, NaN/inf, constant volume,
  empty annotation, annotation outside the mask, labels outside the allowed set.
- **`warning`** = plausible-but-odd, often legitimate: anisotropic spacing,
  spacing/intensity outside the configured range, `int64`/`float64` dtype,
  too many empty slices, fragmented components, bbox touching the border, 4D
  input, sform/qform disagreement.

This split is deliberate: `--strict` is meant for CI gating before a training run,
so only genuine corruption blocks the pipeline; style/portability concerns surface
as warnings without failing the build.

## Geometric conventions (assumed and made explicit)

- **Space.** All geometry is expressed in **nibabel** terms: voxel indices
  `(i, j, k)` and a 4×4 `affine` mapping voxel → world (**RAS+**) coordinates.
- **Orientation.** Reported via `nibabel.aff2axcodes(affine)` (e.g. `('R','A','S')`),
  which names the world axis each voxel axis points toward. Always reported; only
  compared when `expected_orientation` is set.
- **SimpleITK note.** SimpleITK indexes volumes in reversed order `(z, y, x)` and
  uses LPS world coordinates. This module **never mixes** the two conventions — it
  stays entirely in nibabel/RAS space. If you cross into SimpleITK elsewhere in
  your pipeline, the axis order differs.
- **Singularity** is judged on the 3×3 direction/scaling block of the affine: if
  `|det|` ≤ `affine_atol`, the voxel→world map is (near-)non-invertible and the
  volume is flagged. A zero column cannot even be written by nibabel, so the real
  failure mode in practice is a near-collapsed axis.
- **Alignment (pair/triple).** Two volumes can share a shape yet sit at different
  world positions. The check compares full affines with `np.allclose(atol=affine_atol)`
  and reports both the max element-wise difference and the **world-space
  translation offset in mm** — the most intuitive misalignment unit.

## Default thresholds and why

Defaults are **generic and domain-neutral** (no CT/MR assumptions). Override any of
them in `QCConfig` / a config file.

| Field                             | Default       | Rationale |
|-----------------------------------|---------------|-----------|
| `expected_orientation`            | `None`        | Orientation is always reported; only an explicit expectation turns a mismatch into an error. No default forces a convention you might not use. |
| `affine_atol`                     | `1e-4`        | Tight enough to catch sub-voxel shifts and singular affines, loose enough to absorb float round-trips through NIfTI headers. |
| `isotropy_tol`                    | `0.05`        | 5% max/min voxel-edge difference still counts as isotropic — accommodates minor scanner variation without hiding real anisotropy. |
| `spacing_range`                   | `(0.1, 10.0)` mm | Plausible voxel sizes across most modalities; outside is almost always a header/unit bug. |
| `intensity_range`                 | `None`        | Intensity scales are modality-specific; disabled by default to avoid false alarms. Set it to catch missing clipping/normalization. |
| `warn_float64`                    | `True`        | `float32` usually suffices and halves file size; informational only. |
| `empty_slice_bg_value`            | `0.0`         | Voxels ≤ this are background when deciding if a slice is empty. |
| `empty_slice_min_fg_fraction`     | `1e-4`        | A slice with essentially no foreground is "empty"; near-zero so only truly blank slices count. |
| `max_empty_slice_fraction`        | `0.5`         | More than half a volume being blank on an axis signals over-padding or corruption. |
| `allowed_labels`                  | `None` (= `{0,1}`) | Masks/annotations are assumed binary unless you declare a multi-label set. |
| `max_annotation_outside_fraction` | `0.0`         | A lesion should lie entirely within the brain mask; any voxel outside is, by default, an error. Loosen if partial-volume voxels are expected. |
| `min_component_size`              | `1`           | Components below this are flagged as possibly spurious blobs; raise to ignore tiny specks. |
| `max_components`                  | `50`          | Many disconnected components usually means a fragmented/over-thresholded annotation. |
| `warn_bbox_touches_border`        | `True`        | A bounding box touching the volume edge often means clipping/cropping lost part of the object. |
| `max_workers`                     | `4`           | Threads for I/O-bound dataset scans; output is reassembled in sorted order and stays deterministic regardless. |

## Report objects

- `CheckResult(name, status, message, value)` — one check; `value` is the observed
  quantity (JSON-serializable, numpy scalars coerced).
- `Report(target, kind, results, meta)` — one volume/pair/triple; `status` is the
  worst result, `meta` carries shape/dtype/orientation/spacing/intensity context.
- `DatasetReport(root, items, distributions)` — many items plus cross-dataset
  summaries (orientation/spacing/shape/dtype counts and outliers). `worst_first()`
  surfaces the worst items at the top.
- `to_json(report, path=None)` — serialize any of the above; powers `niqc --json`
  for CI, dataset cards and pipelines.

## CLI exit codes

| Code | Condition |
|------|-----------|
| `0`  | Completed; no `error` results, **or** errors present but `--strict` not given. |
| `1`  | `--strict` given and at least one `error` result was found. |
| `2`  | Usage error, bad path, or unreadable config. |

## Dependencies

No new hard dependencies. The module uses only what `nidataset` already ships
(numpy, nibabel, scipy, Pillow, matplotlib). YAML config files are the single
optional extra (`pip install nidataset[yaml]`); **JSON config works with no extra
dependency**, and `qc.example.json` mirrors `qc.example.yaml`.
