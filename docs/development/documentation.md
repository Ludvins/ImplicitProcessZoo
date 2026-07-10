# Documentation workflow

Documentation is written in Markdown below `docs/`, rendered by MkDocs
Material, and published as a latest-only GitHub Pages site.

## Local preview

```bash
python -m pip install -e ".[docs,experiments]"
mkdocs serve
```

Open the local URL printed by MkDocs. The server watches Markdown, docstrings,
and configuration changes.

## Strict build

```bash
mkdocs build --strict
```

Strict mode turns warnings such as missing navigation files, invalid snippets,
or API import failures into build failures. Generated `site/` output is not
committed.

Examples live in `docs/examples/`. Pages embed them with the snippets extension
and CI executes the same files, preventing copied documentation code from
silently drifting.

## Publication

Pull requests build but never deploy. A push to `main` uploads the strict build
as a Pages artifact and deploys it with GitHub's official Pages actions. No
generated `gh-pages` branch is maintained. Repository administrators must set
**Settings → Pages → Source** to **GitHub Actions** once.
