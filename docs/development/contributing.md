# Contributing

Create a focused branch and install the contributor environment:

```bash
python -m pip install -r requirements.txt
```

Keep reusable inference code under `implicit_process_zoo/` and
experiment-specific composition under `experiments/`. New behavior should have
tests, public interfaces should be documented, and generated data/results must
remain outside version control.

Before opening a pull request, run:

```bash
ruff format --check .
ruff check .
python -m pytest -q --cov
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Coverage is enforced across both library and experiment packages. Do not lower
the `fail_under` value in `pyproject.toml`; bug fixes and new public behavior
should normally increase it.

Experiment changes must preserve published protocols unless the scientific
change is explicit, tested, documented, and written to a distinct artifact
root. The ELD code has one canonical corrected protocol and no version switch.
