# Contributing

Create a focused branch, install the development environment, and run the
quality gates before opening a pull request:

```bash
python -m pip install -e ".[experiments,vision,tracking,test]"
ruff format --check .
ruff check .
python -m pytest -q --cov
python -m build
python -m twine check dist/*
```

Keep reusable inference code under `implicit_process_zoo/` and experiment-only
composition under `experiments/`. New behavior should include tests, public
interfaces should be documented, and generated data or results must remain
outside version control.

Coverage is enforced across `implicit_process_zoo` and `experiments`. New work
must not reduce the `fail_under` value in `pyproject.toml`; bug fixes and new
public behavior should normally increase it. Keep methodology-version 1 and 2
ELD artifacts in separate roots and never rewrite published scientific output.
