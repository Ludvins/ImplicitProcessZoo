# Testing

Run the complete suite in the repository environment:

```bash
python -m pytest -q
```

Generate the enforced coverage report:

```bash
python -m pytest -q --cov --cov-report=term-missing --cov-report=xml
```

Quality and publication checks are:

```bash
ruff format --check .
ruff check .
python -m pip check
mkdocs build --strict
python -m build
python -m twine check dist/*
```

The CI matrix runs tests on Python 3.10 and 3.12 using CPU PyTorch. A separate
quality job checks every CLI `--help` command, an ELD synthetic smoke run,
format/lint, and package metadata. Documentation examples are executable files
and are run before every strict site build.
