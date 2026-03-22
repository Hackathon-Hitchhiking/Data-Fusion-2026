from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from lib.layout import COMPETITION_FILES, project_root, resolve_data_dir


ROOT = project_root()
DEFAULT_PROFILE_FILE = ROOT / "configs" / "kaggle_dataset_profiles.json"
DEFAULT_KAGGLE_CLI = ROOT / ".venv" / "bin" / "kaggle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage staged Kaggle datasets for this project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profiles-file",
        type=Path,
        default=DEFAULT_PROFILE_FILE,
        help="JSON file with named dataset profiles.",
    )
    parser.add_argument(
        "--kaggle-cli",
        type=Path,
        default=DEFAULT_KAGGLE_CLI,
        help="Path to the Kaggle CLI executable.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profiles", help="Print configured dataset profiles.")

    init_parser = subparsers.add_parser("init", help="Create stage folders and metadata.")
    init_parser.add_argument("--profile", action="append", default=[], help="Profile name to initialize. Repeatable.")
    init_parser.add_argument("--all", action="store_true", help="Initialize all profiles.")
    init_parser.add_argument("--overwrite-metadata", action="store_true", help="Rewrite dataset-metadata.json.")

    list_parser = subparsers.add_parser("list", help="List staged files for a profile.")
    list_parser.add_argument("--profile", required=True)

    sync_parser = subparsers.add_parser(
        "sync-competition",
        help="Copy the six competition parquet files into the stage dir.",
    )
    sync_parser.add_argument("--profile", default="base_task_dataset")
    sync_parser.add_argument("--data-dir", type=Path, default=None)

    add_parser = subparsers.add_parser("add", help="Copy files into a staged dataset folder.")
    add_parser.add_argument("--profile", required=True)
    add_parser.add_argument(
        "--dest-subdir",
        default="",
        help="Optional subdirectory inside the staged dataset.",
    )
    add_parser.add_argument("files", nargs="+", help="Files to copy into the staged dataset.")

    remove_parser = subparsers.add_parser("remove", help="Delete files from a staged dataset folder.")
    remove_parser.add_argument("--profile", required=True)
    remove_parser.add_argument("--force-core", action="store_true", help="Allow removing core competition files.")
    remove_parser.add_argument("paths", nargs="+", help="Relative staged paths to delete.")

    push_parser = subparsers.add_parser("push", help="Create or version a Kaggle dataset from staged files.")
    push_parser.add_argument("--profile", required=True)
    push_parser.add_argument("-m", "--message", required=True, help="Version notes.")
    push_parser.add_argument(
        "--create-if-missing",
        action="store_true",
        help="Create the dataset if the remote slug does not exist.",
    )
    push_parser.add_argument("--public", action="store_true", help="Create publicly when using --create-if-missing.")
    push_parser.add_argument("--quiet", action="store_true")
    push_parser.add_argument(
        "--delete-old-versions",
        action="store_true",
        help="Pass --delete-old-versions to kaggle datasets version.",
    )
    push_parser.add_argument(
        "--allow-partial-base",
        action="store_true",
        help="Allow pushing the base dataset profile without all six core files staged.",
    )

    return parser.parse_args()


def load_profiles(profiles_file: Path) -> dict[str, dict[str, Any]]:
    if not profiles_file.exists():
        raise FileNotFoundError(f"Profiles file not found: {profiles_file}")
    raw = json.loads(profiles_file.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Profiles file must contain a JSON object.")
    profiles: dict[str, dict[str, Any]] = {}
    for name, profile in raw.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Profile {name!r} must be a JSON object.")
        stage_dir_raw = profile.get("stage_dir")
        if not isinstance(stage_dir_raw, str):
            raise ValueError(f"Profile {name!r} must define string field stage_dir.")
        stage_dir = Path(stage_dir_raw)
        if not stage_dir.is_absolute():
            stage_dir = ROOT / stage_dir
        normalized = dict(profile)
        normalized["stage_dir"] = stage_dir
        profiles[name] = normalized
    return profiles


def require_profile(profiles: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    try:
        return profiles[name]
    except KeyError as exc:
        known = ", ".join(sorted(profiles))
        raise SystemExit(f"Unknown profile {name!r}. Known profiles: {known}") from exc


def ensure_cli_exists(kaggle_cli: Path) -> None:
    if not kaggle_cli.exists():
        raise FileNotFoundError(f"Kaggle CLI not found: {kaggle_cli}")


def dataset_metadata_path(stage_dir: Path) -> Path:
    return stage_dir / "dataset-metadata.json"


def build_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    license_name = str(profile.get("license", "CC0-1.0"))
    metadata: dict[str, Any] = {
        "title": str(profile["title"]),
        "id": str(profile["id"]),
        "licenses": [{"name": license_name}],
    }
    subtitle = profile.get("subtitle")
    description = profile.get("description")
    keywords = profile.get("keywords")
    if subtitle:
        metadata["subtitle"] = str(subtitle)
    if description:
        metadata["description"] = str(description)
    if keywords:
        metadata["keywords"] = list(keywords)
    return metadata


def ensure_profile_initialized(profile: dict[str, Any], *, overwrite_metadata: bool = False) -> Path:
    stage_dir = Path(profile["stage_dir"])
    stage_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_metadata_path(stage_dir)
    if overwrite_metadata or not metadata_path.exists():
        metadata_path.write_text(json.dumps(build_metadata(profile), ensure_ascii=True, indent=2) + "\n")
    return stage_dir


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def file_is_unchanged(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    src_stat = src.stat()
    dst_stat = dst.stat()
    return src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime)


def copy_into_stage(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if file_is_unchanged(src, dst):
        return "unchanged"
    shutil.copy2(src, dst)
    return "copied"


def list_stage_files(stage_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(stage_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.name in {"dataset-metadata.json", "datapackage.json"}:
            continue
        try:
            size_bytes = path.stat().st_size
        except FileNotFoundError:
            continue
        rows.append(
            {
                "path": str(path.relative_to(stage_dir)),
                "size_bytes": size_bytes,
                "size_human": human_size(size_bytes),
            }
        )
    return rows


def is_base_profile(profile_name: str, profile: dict[str, Any]) -> bool:
    return profile_name == "base_task_dataset" or str(profile["id"]).endswith("data-fusion-2026-2-task-dataset")


def validate_base_stage_complete(profile_name: str, profile: dict[str, Any], allow_partial_base: bool) -> None:
    if not is_base_profile(profile_name, profile) or allow_partial_base:
        return
    stage_dir = Path(profile["stage_dir"])
    missing = [file_name for file_name in COMPETITION_FILES if not (stage_dir / file_name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            "Refusing to push the base dataset profile without all six core files staged. "
            f"Missing: {missing_text}. Run sync-competition first or pass --allow-partial-base."
        )


def safe_relative_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise SystemExit(f"Expected a relative staged path, got absolute path: {raw_path}")
    if ".." in path.parts:
        raise SystemExit(f"Refusing path with '..': {raw_path}")
    return path


def dataset_exists(kaggle_cli: Path, dataset_id: str) -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [str(kaggle_cli), "datasets", "metadata", "-p", tmpdir, dataset_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return result.returncode == 0


def run_kaggle(kaggle_cli: Path, args: list[str]) -> None:
    subprocess.run([str(kaggle_cli), *args], check=True)


def cmd_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    payload = {
        name: {
            "id": profile["id"],
            "title": profile["title"],
            "stage_dir": str(profile["stage_dir"]),
            "notes": profile.get("notes", ""),
        }
        for name, profile in profiles.items()
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_init(args: argparse.Namespace, profiles: dict[str, dict[str, Any]]) -> None:
    names = sorted(profiles) if args.all else args.profile
    if not names:
        raise SystemExit("Pass --all or at least one --profile.")
    for name in names:
        profile = require_profile(profiles, name)
        stage_dir = ensure_profile_initialized(profile, overwrite_metadata=args.overwrite_metadata)
        print(json.dumps({"profile": name, "stage_dir": str(stage_dir), "metadata": str(dataset_metadata_path(stage_dir))}))


def cmd_list(profile_name: str, profile: dict[str, Any]) -> None:
    stage_dir = ensure_profile_initialized(profile)
    print(
        json.dumps(
            {
                "profile": profile_name,
                "dataset_id": profile["id"],
                "stage_dir": str(stage_dir),
                "files": list_stage_files(stage_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_sync_competition(profile_name: str, profile: dict[str, Any], data_dir: Path | None) -> None:
    stage_dir = ensure_profile_initialized(profile)
    resolved_data_dir = resolve_data_dir(data_dir, required_files=COMPETITION_FILES)
    rows: list[dict[str, Any]] = []
    for file_name in COMPETITION_FILES:
        src = resolved_data_dir / file_name
        dst = stage_dir / file_name
        rows.append(
            {
                "file": file_name,
                "status": copy_into_stage(src, dst),
                "size_human": human_size(src.stat().st_size),
            }
        )
    print(
        json.dumps(
            {
                "profile": profile_name,
                "dataset_id": profile["id"],
                "stage_dir": str(stage_dir),
                "copied": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_add(profile_name: str, profile: dict[str, Any], files: list[str], dest_subdir: str) -> None:
    stage_dir = ensure_profile_initialized(profile)
    subdir = safe_relative_path(dest_subdir) if dest_subdir else Path()
    rows: list[dict[str, Any]] = []
    for raw_path in files:
        src = Path(raw_path).expanduser().resolve()
        if not src.exists() or not src.is_file():
            raise SystemExit(f"File not found: {src}")
        dst = stage_dir / subdir / src.name
        rows.append(
            {
                "src": str(src),
                "dst": str(dst.relative_to(stage_dir)),
                "status": copy_into_stage(src, dst),
                "size_human": human_size(src.stat().st_size),
            }
        )
    print(
        json.dumps(
            {
                "profile": profile_name,
                "dataset_id": profile["id"],
                "stage_dir": str(stage_dir),
                "added": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_remove(profile_name: str, profile: dict[str, Any], paths: list[str], force_core: bool) -> None:
    stage_dir = ensure_profile_initialized(profile)
    removed: list[str] = []
    for raw_path in paths:
        rel_path = safe_relative_path(raw_path)
        if is_base_profile(profile_name, profile) and rel_path.name in COMPETITION_FILES and not force_core:
            raise SystemExit(
                f"Refusing to remove core competition file {rel_path.name!r} from base profile without --force-core."
            )
        target = stage_dir / rel_path
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append(str(rel_path))
    print(
        json.dumps(
            {
                "profile": profile_name,
                "dataset_id": profile["id"],
                "stage_dir": str(stage_dir),
                "removed": removed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_push(
    kaggle_cli: Path,
    profile_name: str,
    profile: dict[str, Any],
    *,
    message: str,
    create_if_missing: bool,
    public: bool,
    quiet: bool,
    delete_old_versions: bool,
    allow_partial_base: bool,
) -> None:
    stage_dir = ensure_profile_initialized(profile)
    validate_base_stage_complete(profile_name, profile, allow_partial_base)
    if not list_stage_files(stage_dir):
        raise SystemExit(f"Stage dir {stage_dir} has no data files to upload.")

    remote_exists = dataset_exists(kaggle_cli, str(profile["id"]))
    if remote_exists:
        cmd = ["datasets", "version", "-p", str(stage_dir), "-m", message, "-t"]
        if quiet:
            cmd.append("-q")
        if delete_old_versions:
            cmd.append("-d")
    else:
        if not create_if_missing:
            raise SystemExit(
                f"Remote dataset {profile['id']} does not exist. Re-run with --create-if-missing to create it."
            )
        cmd = ["datasets", "create", "-p", str(stage_dir), "-t"]
        if public:
            cmd.append("-u")
        if quiet:
            cmd.append("-q")

    print(
        json.dumps(
            {
                "profile": profile_name,
                "dataset_id": profile["id"],
                "stage_dir": str(stage_dir),
                "remote_exists": remote_exists,
                "command": [str(kaggle_cli), *cmd],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    run_kaggle(kaggle_cli, cmd)


def main() -> None:
    args = parse_args()
    ensure_cli_exists(args.kaggle_cli)
    profiles = load_profiles(args.profiles_file)

    if args.command == "profiles":
        cmd_profiles(profiles)
        return

    if args.command == "init":
        cmd_init(args, profiles)
        return

    if args.command == "list":
        profile = require_profile(profiles, args.profile)
        cmd_list(args.profile, profile)
        return

    if args.command == "sync-competition":
        profile = require_profile(profiles, args.profile)
        cmd_sync_competition(args.profile, profile, args.data_dir)
        return

    if args.command == "add":
        profile = require_profile(profiles, args.profile)
        cmd_add(args.profile, profile, args.files, args.dest_subdir)
        return

    if args.command == "remove":
        profile = require_profile(profiles, args.profile)
        cmd_remove(args.profile, profile, args.paths, args.force_core)
        return

    if args.command == "push":
        profile = require_profile(profiles, args.profile)
        cmd_push(
            args.kaggle_cli,
            args.profile,
            profile,
            message=args.message,
            create_if_missing=args.create_if_missing,
            public=args.public,
            quiet=args.quiet,
            delete_old_versions=args.delete_old_versions,
            allow_partial_base=args.allow_partial_base,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
