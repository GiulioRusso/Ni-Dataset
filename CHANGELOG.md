# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps may add features).

## [0.9.0] - Unreleased

### Added
- `nidataset.pipeline`: config-driven pipelines. Describe a dataset and ordered
  steps in YAML (`pip install "nidataset[yaml]"`) or JSON, run with `nii run
  pipeline.yaml` or `nid.run_pipeline(path)`. The whole config is validated
  before anything runs; each run writes `run_manifest.json` (nidataset version,
  config hash, per-step timing and outputs). `nii run --only STEP[,STEP]` reruns
  selected steps.
- `examples/ct_pipeline/`: end-to-end CT example (QC, slices, annotations,
  registration, MIP, resampling) with a synthetic data generator, so it runs
  in seconds without patient data.
- Tests for slices, preprocessing, analysis, transforms, volume, both CLIs and
  the pipeline; world-coordinate invariant tests for rotate/flip/crop/crop-and-pad
  and resampling. Tests needing FSL are marked `fsl` and skip without it.
- GitHub Actions CI (Python 3.9–3.13, Ubuntu and macOS, ruff, coverage gate 70%)
  and a tag-triggered release workflow using PyPI Trusted Publishing.
- `CITATION.cff`, `CONTRIBUTING.md`, this changelog.

### Fixed
- `draw_3D_boxes`: `MAX` coordinates are now inclusive, matching `extract_annotations`,
  `extract_bounding_boxes` and `dataset_annotations_info` (boxes were drawn one voxel
  short on every axis), and a negative `MIN` no longer wraps around to the opposite
  side of the volume.
- `register_CTA` / `register_CTA_dataset`: `learning_rate` was silently ignored (the
  rate was always re-estimated). It now defaults to `None` (estimate, as before) and
  an explicit value is honoured. New `random_seed` (default `42`) seeds the metric
  sampling, which was previously seeded from the wall clock, so registrations are
  now reproducible run to run.
- `dicom_to_nifti`: slices are now ordered along the slice normal before the
  volume is assembled. SimpleITK skips that sort on its first call in a process,
  which produced a volume shifted in world space for series whose files are not
  stored in normal order (e.g. after `nifti_to_dicom` of a left-handed affine).
- `dicom_to_nifti` / `nifti_to_dicom` no longer leave the process in the `C`
  locale (set by GDCM), which made Python's default text encoding ASCII for the
  rest of the run.

### Changed
- Minimum Python is now 3.9 (3.8 is end-of-life).
- `QCConfig.load` shares its JSON/YAML loader with the pipeline; the missing-PyYAML
  error now suggests `pip install "nidataset[yaml]"`.
- Codebase linted and formatted with ruff (`E, F, I, B, UP, RUF`).

## [0.8.0] - 2026-07-08

### Added
- `nii` CLI: rotate, flip, crop, crop-to-content, rescale, DICOM conversion, preview.
- `nidataset.qc` quality-control module and `niqc` CLI.

[0.9.0]: https://github.com/GiulioRusso/Ni-Dataset/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/GiulioRusso/Ni-Dataset/compare/v0.7.0...v0.8.0
