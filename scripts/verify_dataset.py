#!/usr/bin/env python3
"""Verify released files against the public SHA-256 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(root: Path, dataset: str, require_all: bool) -> tuple[int, int]:
    manifest_path = root / "metadata" / dataset / "manifest.csv"
    data_root = root / "data" / dataset
    missing = []
    mismatched = []
    expected_paths = set()
    checked = 0

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        relative = Path(row["relative_path"])
        expected_paths.add(relative.as_posix())
        path = data_root / relative
        should_exist = row["available_in_repository"].lower() == "true" or require_all
        if not path.is_file():
            if should_exist:
                missing.append(relative.as_posix())
            continue
        checked += 1
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != int(row["size_bytes"]) or actual_hash != row["sha256"]:
            mismatched.append(relative.as_posix())

    actual_paths = {
        path.relative_to(data_root).as_posix()
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    unexpected = sorted(actual_paths - expected_paths)

    if missing or mismatched or unexpected:
        print(
            f"FAIL {dataset}: checked={checked}, missing={len(missing)}, "
            f"mismatched={len(mismatched)}, unexpected={len(unexpected)}"
        )
        for label, values in (
            ("missing", missing),
            ("mismatched", mismatched),
            ("unexpected", unexpected),
        ):
            for value in values[:10]:
                print(f"  {label}: {value}")
        return 1, checked

    note = ""
    if checked < len(rows):
        note = f" ({len(rows) - checked} licensed files intentionally absent)"
    print(f"OK   {dataset}: {checked}/{len(rows)} files verified{note}")
    return 0, checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument(
        "--require-plant-pathology",
        action="store_true",
        help="require the locally reconstructed Plant Pathology images",
    )
    args = parser.parse_args()

    failures = 0
    for dataset in ("PlantVillage", "AppleLeaf9", "PlantCity"):
        result, _ = verify_dataset(args.root, dataset, True)
        failures += result
    result, _ = verify_dataset(
        args.root, "PlantPathology", args.require_plant_pathology
    )
    failures += result
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

