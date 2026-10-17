# Contributing

1. **Setup**: `git clone https://github.com/GiulioRusso/Ni-Dataset && cd Ni-Dataset && pip install -e ".[dev,yaml]"`
2. **Test**: `pytest --cov=nidataset` (tests marked `fsl` run only when FSL's `bet`/`fslmaths` are on `PATH`).
3. **Lint**: `ruff check --fix nidataset tests examples && ruff format nidataset tests examples` (CI runs both, pinned ruff version in `.github/workflows/ci.yml`).
4. **Test data**: synthetic volumes only (see `tests/conftest.py`); never commit patient data.
5. **Geometric ops**: add a world-coordinate invariant test (`tests/test_affine_invariants.py`).
6. **Changelog**: add a line under `Unreleased` in `CHANGELOG.md`.
7. **PRs**: CI must be green on all Python/OS cells.
8. **Release**: bump `nidataset/_version.py`, date the `CHANGELOG.md` entry, update `CITATION.cff` `version`.
9. Tag and push: `git tag v0.9.1 && git push origin v0.9.1`; `release.yml` builds and publishes to PyPI (Trusted Publishing).
10. Check the release on https://pypi.org/project/nidataset/.
