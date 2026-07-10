# Contributing

Create a focused branch, install the development environment, and run the
quality gates before opening a pull request:

```bash
python -m pip install -e ".[experiments,vision,tracking,test]"
ruff format --check .
ruff check .
python -m pytest -q
```

Keep reusable inference code under `implicit_process_zoo/` and experiment-only
composition under `experiments/`. New behavior should include tests, public
interfaces should be documented, and generated data or results must remain
outside version control.
