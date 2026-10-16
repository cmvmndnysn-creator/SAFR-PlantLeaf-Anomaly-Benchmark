#!/usr/bin/env python3
"""Materialize the selected Plant Pathology subset from an authorized download."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the benchmark's Plant Pathology 2021 subset from "
            "an official download obtained after accepting the Kaggle rules."
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hardlink", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = args.manifest or repo_root / "metadata/PlantPathology/manifest.csv"

    candidates: dict[str, list[Path]] = defaultdict(list)
    for path in args.source.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates[path.name].append(path)

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    copied = 0
    for row in rows:
        name = Path(row["relative_path"]).name
        matches = candidates.get(name, [])
        source = None
        for candidate in matches:
            if candidate.stat().st_size == int(row["size_bytes"]):
                if sha256(candidate) == row["sha256"]:
                    source = candidate
                    break
        if source is None:
            raise FileNotFoundError(
                f"Could not find an authorized source file matching {name} "
                f"and SHA-256 {row['sha256']}"
            )

        destination = args.output / row["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not args.force:
                raise FileExistsError(
                    f"Destination exists: {destination}; pass --force to replace it"
                )
            destination.unlink()
        if args.hardlink:
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)
        copied += 1

    print(f"Materialized and verified {copied} Plant Pathology images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

