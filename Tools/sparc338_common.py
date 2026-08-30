"""Small shared I/O, provenance and safe-output helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: Iterable[dict], fields: Iterable[str]) -> None:
    rows = list(rows)
    fields = tuple(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.resolve()): sha256_file(path)
        for path in sorted(root.rglob("*"), key=str)
        if path.is_file()
    }


def source_record(path: Path, include_sha256: bool = False) -> dict:
    stat = path.stat()
    row = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        row["sha256"] = sha256_file(path)
    return row


def make_staging_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))


def commit_directory(staging: Path, target: Path) -> None:
    """Replace target only after a complete staging directory has been built.

    A sibling rollback directory is retained until the new directory is in place.
    This also removes stale cycle/event files from earlier runs.
    """
    staging = staging.resolve()
    target = target.resolve()
    if staging.parent != target.parent:
        raise ValueError("Staging and target directories must be siblings")
    rollback = target.with_name(f".{target.name}.rollback")
    if rollback.exists():
        shutil.rmtree(rollback)
    had_target = target.exists()
    if had_target:
        os.replace(target, rollback)
    try:
        os.replace(staging, target)
    except Exception:
        if had_target and rollback.exists() and not target.exists():
            os.replace(rollback, target)
        raise
    if rollback.exists():
        shutil.rmtree(rollback)


def cleanup_staging(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
