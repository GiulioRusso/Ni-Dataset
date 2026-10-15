"""
nidataset.pipeline — config-driven pipelines over the nidataset API.

Describe a dataset and an ordered list of steps in YAML or JSON, then run it:

>>> import nidataset as nid
>>> nid.run_pipeline("pipeline.yaml")

or from the shell::

    nii run pipeline.yaml

Each step names an ``op`` (see :data:`OPS`) and passes the remaining keys as
kwargs to the underlying nidataset function. The whole config is validated
before anything runs, and ``<output>/run_manifest.json`` records the nidataset
version, config hash, per-step timing and outputs.
"""

from .registry import OPS
from .runner import PipelineError, plan, run_pipeline
from .schema import PipelineConfig, PipelineConfigError, Step

__all__ = ["OPS", "PipelineConfig", "PipelineConfigError", "PipelineError", "Step", "plan", "run_pipeline"]
