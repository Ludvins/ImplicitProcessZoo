"""Clean W&B project runs and local generated logs for APFSVI experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "outputs" / "wandb_cleanup" / "apfsvi_runs_manifest.json"


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO.resolve())
        return True
    except ValueError:
        return False


def collect_local_log_paths() -> list[Path]:
    paths: list[Path] = []
    for candidate in (
        REPO / "outputs" / "wandb_logs",
        REPO / "outputs" / "wandb_runtime",
    ):
        if candidate.exists():
            paths.append(candidate)
    results_dir = REPO / "results"
    if results_dir.exists():
        paths.extend(path for path in results_dir.rglob("wandb") if path.is_dir())
    return sorted(set(paths), key=lambda p: str(p).lower())


def clean_local_logs(yes: bool) -> None:
    paths = collect_local_log_paths()
    print(f"Local log/cache paths: {len(paths)}")
    for path in paths:
        print(path)
    if not yes:
        print("Dry run only. Re-run with --yes to delete these local paths.")
        return
    for path in paths:
        if not _inside_repo(path):
            raise RuntimeError(f"Refusing to delete path outside repo: {path}")
        if path.exists():
            shutil.rmtree(path)
            print(f"deleted {path}")


def write_wandb_manifest(entity: str, project: str, manifest: Path) -> None:
    import wandb

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    records = []
    for run in runs:
        records.append(
            {
                "entity": entity,
                "project": project,
                "id": run.id,
                "name": run.name,
                "group": run.group,
                "state": run.state,
                "created_at": str(run.created_at),
                "url": run.url,
            }
        )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} W&B runs to {manifest}")


def delete_wandb_manifest(entity: str, project: str, manifest: Path, yes: bool) -> None:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Manifest must contain a JSON list.")
    for record in records:
        if record.get("entity") != entity or record.get("project") != project:
            raise ValueError(
                "Manifest contains a run outside the requested entity/project: "
                f"{record}"
            )
    print(f"Manifest contains {len(records)} runs from {entity}/{project}.")
    if not yes:
        print("Dry run only. Re-run with --delete-wandb --yes to delete these runs.")
        return

    import wandb

    api = wandb.Api()
    for idx, record in enumerate(records, start=1):
        run_id = record["id"]
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"[{idx}/{len(records)}] deleting {run_id}: {record.get('name')}")
        run.delete()
    print(f"Deleted {len(records)} W&B runs from {entity}/{project}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="ludvins")
    parser.add_argument("--project", default="apfsvi")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write-wandb-manifest", action="store_true")
    parser.add_argument("--delete-wandb", action="store_true")
    parser.add_argument("--clean-local-logs", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if args.write_wandb_manifest:
        write_wandb_manifest(args.entity, args.project, args.manifest)
    if args.delete_wandb:
        delete_wandb_manifest(args.entity, args.project, args.manifest, args.yes)
    if args.clean_local_logs:
        clean_local_logs(args.yes)
    if not (args.write_wandb_manifest or args.delete_wandb or args.clean_local_logs):
        parser.error(
            "Choose at least one action: --write-wandb-manifest, "
            "--delete-wandb, or --clean-local-logs."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
