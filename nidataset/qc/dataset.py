"""
Dataset-level QC: per-item checks plus cross-dataset coherence.

Accepts a folder of loose NIfTI files, or a CSV listing pairs/triples (the format
used for detection). Each item is checked independently (lazily — one volume in
memory at a time), then cross-dataset distributions are summarized: a dataset
where 287 volumes are RAS and 13 are LAS has a silent orientation bug the
per-file view never surfaces.

Reads are lazy and may run on a thread pool (I/O-bound ``nib.load``), but the
returned :class:`DatasetReport` is always deterministic and sorted.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

from .._helpers import list_nifti_files, is_nifti, logger
from .checks import check_volume
from .config import QCConfig
from .pairs import check_pair, check_triple
from .report import Report, DatasetReport

# CSV column names recognised for pair/triple manifests (case-insensitive).
_IMAGE_COLS = ("image", "img", "ct", "cta", "volume")
_MASK_COLS = ("mask", "brain", "brain_mask")
_ANN_COLS = ("annotation", "ann", "label", "lesion", "bbox", "gt")


def _pick(header: List[str], candidates: Tuple[str, ...]) -> Optional[int]:
    """Return the index of the first header column matching *candidates*."""
    lower = [h.strip().lower() for h in header]
    for cand in candidates:
        if cand in lower:
            return lower.index(cand)
    return None


def _scan_csv(csv_path: str, config: QCConfig) -> List[Callable[[], Report]]:
    """Build per-row check thunks from a CSV manifest of pairs/triples."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"CSV '{csv_path}' is empty.")

    header = rows[0]
    has_header = any(any(c.isalpha() for c in cell) and not is_nifti(cell) for cell in header)
    if has_header:
        i_img = _pick(header, _IMAGE_COLS)
        i_mask = _pick(header, _MASK_COLS)
        i_ann = _pick(header, _ANN_COLS)
        if i_img is None or i_mask is None:
            # Fall back to positional: col0=image, col1=mask, col2=annotation.
            i_img, i_mask, i_ann = 0, 1, (2 if len(header) > 2 else None)
        data_rows = rows[1:]
    else:
        i_img, i_mask = 0, 1
        i_ann = 2 if len(header) > 2 else None
        data_rows = rows

    thunks: List[Callable[[], Report]] = []
    for row in data_rows:
        if not row or all(not cell.strip() for cell in row):
            continue
        img = row[i_img].strip()
        mask = row[i_mask].strip()
        ann = row[i_ann].strip() if (i_ann is not None and i_ann < len(row) and row[i_ann].strip()) else None
        if ann:
            thunks.append(lambda img=img, mask=mask, ann=ann: check_triple(img, mask, ann, config))
        else:
            thunks.append(lambda img=img, mask=mask: check_pair(img, mask, config))
    return thunks


def _scan_folder(folder: str, config: QCConfig) -> List[Callable[[], Report]]:
    """Build per-file ``check_volume`` thunks for every NIfTI in *folder* (sorted)."""
    files = list_nifti_files(folder)
    return [lambda p=os.path.join(folder, f): check_volume(p, config) for f in files]


def _distributions(items: List[Report]) -> dict:
    """Summarize orientation / spacing / shape / dtype across volume items."""
    orient = Counter()
    spacing = Counter()
    shape = Counter()
    dtype = Counter()
    outliers = {"orientation": [], "dtype": []}

    metas = [(it.target, it.meta) for it in items if "orientation" in it.meta or "dtype" in it.meta]
    for _, meta in metas:
        if "orientation" in meta:
            orient[meta["orientation"]] += 1
        if "spacing" in meta:
            spacing[tuple(round(z, 2) for z in meta["spacing"])] += 1
        if "shape" in meta:
            shape[tuple(meta["shape"])] += 1
        if "dtype" in meta:
            dtype[meta["dtype"]] += 1

    # Outliers = items not sharing the majority value (the minority worth listing).
    def _outliers(key: str, counter: Counter) -> List[str]:
        if len(counter) <= 1:
            return []
        majority = counter.most_common(1)[0][0]
        return [target for target, meta in metas if key in meta and meta[key] != majority]

    outliers["orientation"] = _outliers("orientation", orient)
    outliers["dtype"] = _outliers("dtype", dtype)

    return {
        "orientation": dict(orient),
        "spacing": {str(k): v for k, v in spacing.items()},
        "shape": {str(k): v for k, v in shape.items()},
        "dtype": dict(dtype),
        "outliers": outliers,
    }


def check_dataset(path: str, config: Optional[QCConfig] = None,
                  max_workers: Optional[int] = None) -> DatasetReport:
    """
    Run QC over a whole dataset and summarize cross-dataset coherence.

    *path* is auto-detected: a folder of NIfTI files runs :func:`check_volume` on
    each; a ``.csv`` manifest runs :func:`check_pair` / :func:`check_triple` per
    row (columns ``image,mask[,annotation]``, with or without a header).

    :param path:        Folder or ``.csv`` manifest.
    :param config:      Thresholds; defaults to :class:`QCConfig`.
    :param max_workers: Thread workers (overrides ``config.max_workers``). ``1``
        disables parallelism. Output order is deterministic regardless.

    :returns: A :class:`DatasetReport` with per-item reports (sorted) and
        ``distributions`` (orientation/spacing/shape/dtype counts + outliers).

    Example
    -------
    >>> from nidataset.qc import check_dataset
    >>> ds = check_dataset("scans/")
    >>> ds.counts(), ds.distributions["orientation"]
    """
    config = config or QCConfig()
    workers = max_workers if max_workers is not None else config.max_workers

    if path.lower().endswith(".csv"):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV manifest not found: '{path}'")
        thunks = _scan_csv(path, config)
    elif os.path.isdir(path):
        thunks = _scan_folder(path, config)
    else:
        raise FileNotFoundError(f"'{path}' is neither a folder nor a .csv manifest.")

    # Run thunks; results reassembled in submission order for determinism.
    if workers and workers > 1 and len(thunks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            items: List[Report] = list(pool.map(lambda t: t(), thunks))
    else:
        items = [t() for t in thunks]

    distributions = _distributions(items)
    logger.info("check_dataset(%s): %d items, status=%s", path, len(items),
                DatasetReport(root=path, items=items).status)
    return DatasetReport(root=path, items=items, distributions=distributions)
