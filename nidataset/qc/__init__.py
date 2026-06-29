"""
nidataset.qc — quality control and validation for NIfTI datasets.

Answers a question no viewer does: *is this dataset geometrically coherent and
trustworthy, or is something silently poisoning training?* It hunts the bugs that
never raise — unexpected orientation (LAS vs RAS), a mask shifted a few voxels
from its image, empty annotations, anisotropic spacing, all-black border slices,
non-portable ``int64`` data, NaN/inf — and reports them as inspectable,
serializable objects.

Quick start
-----------
>>> import nidataset as nid
>>> rep = nid.qc.check_volume("scan.nii.gz")
>>> rep.status                      # 'ok' | 'warning' | 'error'
>>> [r.name for r in rep.issues()]  # only the problems
>>> nid.qc.to_json(rep, "report.json")

Pair / triple coherence (the high-value path for detection/segmentation):

>>> rep = nid.qc.check_pair("ct.nii.gz", "brain_mask.nii.gz")
>>> rep = nid.qc.check_triple("ct.nii.gz", "brain.nii.gz", "lesion.nii.gz")

Whole dataset + custom thresholds:

>>> cfg = nid.qc.QCConfig(expected_orientation="RAS", affine_atol=1e-3)
>>> ds = nid.qc.check_dataset("scans/", config=cfg)
>>> ds.distributions["orientation"]   # e.g. {'RAS': 287, 'LAS': 13}

The CLI ``niqc`` wraps the same functions; see ``niqc --help``.
"""

from .config import QCConfig
from .report import (
    CheckResult,
    Report,
    DatasetReport,
    to_json,
    OK,
    WARNING,
    ERROR,
)
from .checks import check_volume
from .pairs import check_pair, check_triple
from .dataset import check_dataset
from .thumbnails import thumbnail_grid

__all__ = [
    "QCConfig",
    "CheckResult",
    "Report",
    "DatasetReport",
    "to_json",
    "OK",
    "WARNING",
    "ERROR",
    "check_volume",
    "check_pair",
    "check_triple",
    "check_dataset",
    "thumbnail_grid",
]
