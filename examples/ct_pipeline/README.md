# CT pipeline example (`nii run`)

End-to-end CT dataset pipeline — QC, slice/mask/annotation extraction, annotation
overlay, registration to a template, MIP, resampling — driven by one config file.
This replaces the former [CT-Manager](https://github.com/GiulioRusso/CT-manager) repo.

```bash
pip install "nidataset[yaml]"   # YAML configs; .json configs need no extra
python make_synthetic.py        # 2 tiny synthetic cases + template in ./data
nii run pipeline.yaml           # writes ./output and ./output/run_manifest.json
```

Runs in a few seconds; no real patient data involved.

## Config

```yaml
dataset:                 # named inputs, relative to the config file
  images: data/images
  masks: data/masks
output: output           # run folder
steps:                   # run in order
  - op: extract_slices
    views: [axial, coronal, sagittal]   # optional: one run per view
    nii_folder: "{images}"              # other keys = kwargs of the op
    saving_mode: view
```

- Every key besides `op`, `name`, `views` is passed to the underlying nidataset
  function, so that function's docs are the reference for its options.
- Strings may use `{<dataset key>}`, `{output}` and `{view}`. Quote them in YAML.
- Each step writes to `{output}/<name>` (`name` defaults to `op`; set it when an op
  appears twice). Later steps read earlier outputs via `{output}/<name>/...`.
- The whole config is validated before anything runs: unknown ops, bad kwargs,
  unknown placeholders, missing input paths and missing FSL are all reported at once.
- Rerun part of it with `nii run pipeline.yaml --only register,register_mask`
  (outputs of skipped steps must already exist). This replaces CT-Manager's `--task`.

Full reference: [Pipelines documentation](https://giuliorusso.github.io/Ni-Dataset/pipeline/).

| op | nidataset function |
|----|--------------------|
| `extract_slices` | `extract_slices_dataset` |
| `extract_annotations` | `extract_annotations_dataset` |
| `draw_annotations` | `draw_2D_annotations` |
| `skull_strip` | `skull_CTA_dataset` (needs FSL: `bet`, `fslmaths`) |
| `register` | `register_CTA_dataset` |
| `register_mask` | `register_mask_dataset` |
| `register_annotation` | `register_annotation_dataset` |
| `mip` | `mip_dataset` |
| `resample` | `resampling_dataset` |
| `qc_volume` | `qc.check_volume` → `<case>_qc.json` (`nii_path`, optional `config_path`) |
| `qc_dataset` | `qc.check_dataset` → `qc_dataset.json` (`dataset_path`, optional `config_path`) |

Same thing from Python:

```python
import nidataset as nid

manifest = nid.run_pipeline("pipeline.yaml")
for step in manifest["steps"]:
    print(step["step"], step["view"], step["status"], step["seconds"])
```

`run_manifest.json` records the nidataset and Python versions, a SHA-256 of the
config, start/end time and, per step, the resolved kwargs, timing, return value
and output folder. A failing step is recorded (`status: failed`, `error`) before
the run stops.

### Using your own data

Point `dataset:` at your folders. Naming conventions the registration ops rely on:
image `<case>.nii.gz`, brain mask `<case>_mask.nii.gz`, annotation `<case>_bbox.nii.gz`.
For a real template (e.g. MNI152) set `template` / `template_mask` to its brain and
brain-mask files and drop the `number_iterations` / `metric_sampling_percentage`
overrides (they are tuned for the tiny synthetic volumes).

**Skull stripping** needs FSL on `PATH`. Run from a terminal that sources FSL's
profile (IDEs often don't). Paths with spaces are fine: FSL works in a temp dir.

## Quality control

QC catches silent dataset bugs (wrong orientation, shifted masks, NaN, anisotropic
spacing) before they poison training. In a pipeline, use the `qc_volume` /
`qc_dataset` steps; standalone, use `niqc` or the Python API.

### Single volume

```bash
niqc data/images/case01.nii.gz
```

```python
import nidataset as nid

report = nid.qc.check_volume("ct.nii.gz")
print(report.status)  # "ok", "warning", or "error"
for issue in report.issues():
    print(f"{issue.name}: {issue.message}")
```

### Dataset folder (cross-file coherence)

```bash
niqc data/images --json qc_dataset.json
```

```python
ds = nid.qc.check_dataset("scans/")
print(len(ds.items), ds.status)
print(ds.distributions["orientation"])  # e.g. {'RAS': 40, 'LAS': 2}
for path in ds.distributions["outliers"]["orientation"]:
    print(f"Orientation mismatch: {path}")
```

### Image ↔ mask pairs, image ↔ mask ↔ annotation triples

A mask shifted 3 mm in world space with matching shape never crashes, it just
poisons training.

```bash
niqc --pair data/images/case01.nii.gz data/masks/case01_mask.nii.gz
niqc --triple ct.nii.gz brain.nii.gz lesion.nii.gz --thumbnails qc/
```

```python
rep = nid.qc.check_pair("ct.nii.gz", "brain_mask.nii.gz")
if "affine_match_mask" in [r.name for r in rep.issues()]:
    print("ALERT: mask misaligned in world space!")
```

Triples check shape match, affine alignment, annotation non-empty, annotation
inside the mask and valid labels (binary by default).

### Common silent bugs

```python
import numpy as np
import nibabel as nib

data = np.random.rand(64, 64, 64).astype(np.float32)
data[5, 5, 5] = np.nan
nib.save(nib.Nifti1Image(data, np.eye(4)), "bad.nii.gz")
print(nid.qc.check_volume("bad.nii.gz").issues())  # finite_values: 1 NaN ...
```

- **LAS vs RAS mix**: `check_dataset(...).distributions["orientation"]` → `{'RAS': 287, 'LAS': 13}`.
- **Anisotropic spacing**: `spacing_isotropy` warning when `max/min - 1 > isotropy_tol`.

### Custom thresholds

`qc.json`:

```json
{
  "expected_orientation": "RAS",
  "affine_atol": 0.001,
  "max_empty_slice_fraction": 0.3,
  "isotropy_tol": 0.1
}
```

```bash
niqc scans/ --config qc.json
```

or in a pipeline step: `config_path: qc.json`; or `nid.qc.QCConfig.load("qc.json")`.

### CI

`niqc scans/ --strict` exits `1` if any check is an error (`0` otherwise, `2` on
usage errors). `nii run` exits `2` on an invalid config or a failed step.

### What gets checked

- **Geometry**: affine non-singular, orientation, spacing isotropy, 4D volumes.
- **Data**: NaN / Inf, dtype (int64 / float64 warned), all-zero volumes, empty slices.
- **Pairs / triples**: shape match, world-space affine alignment, annotation
  non-empty and inside the mask, valid labels.
