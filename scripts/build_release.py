#!/usr/bin/env python3
"""Build the public release from the four processed experiment directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FIELDS = [
    "dataset",
    "relative_path",
    "split",
    "shot",
    "crop",
    "category",
    "binary_label",
    "source_image_id",
    "sha256",
    "size_bytes",
    "available_in_repository",
]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def classify(dataset: str, relative: Path) -> tuple[str, str, str, str, int]:
    parts = relative.parts
    if dataset == "PlantVillage":
        if parts[0].startswith("train_"):
            return "train", parts[0].split("_")[1], parts[1].split("___")[0], "healthy", 0
        category = parts[3]
        return "test", "", parts[1], category, 0 if category == "good" else 1
    if dataset == "AppleLeaf9":
        if parts[0].startswith("train_"):
            return "train", parts[0].split("_")[1], "Apple", "healthy", 0
        category = parts[1]
        return "test", "", "Apple", category, 0 if category == "good" else 1
    if dataset == "PlantCity":
        if parts[0] == "train":
            return "train", parts[1].split("_")[1], parts[2], "healthy", 0
        category = parts[3]
        return "test", "", parts[1], category, 0 if category == "good" else 1
    raise ValueError(dataset)


def copy_and_manifest(dataset: str, source: Path, destination: Path) -> list[dict[str, str]]:
    rows = []
    for path in image_files(source):
        relative = path.relative_to(source)
        if any(part.startswith("_backup_") for part in relative.parts):
            continue
        if relative.parts[:2] == ("metadata", "removed_label_conflicts"):
            continue
        split, shot, crop, category, label = classify(dataset, relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rows.append(
            {
                "dataset": dataset,
                "relative_path": relative.as_posix(),
                "split": split,
                "shot": shot,
                "crop": crop,
                "category": category,
                "binary_label": str(label),
                "source_image_id": path.stem,
                "sha256": file_hash(target),
                "size_bytes": str(target.stat().st_size),
                "available_in_repository": "true",
            }
        )
    return rows


def plant_pathology_manifest(source: Path) -> list[dict[str, str]]:
    manifest = source / "metadata/split_manifest.csv"
    rows = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for original in csv.DictReader(handle):
            relative = Path(original["target_path"])
            path = source / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = file_hash(path)
            if actual_hash != original["sha256"]:
                raise ValueError(f"Hash mismatch in source metadata: {path}")
            split = "train" if relative.parts[0] == "train" else "test"
            shot = original["shot"] if split == "train" else ""
            role = original["role"]
            category = "healthy" if split == "train" else original["category"]
            label = 0 if "good" in role else 1
            rows.append(
                {
                    "dataset": "PlantPathology",
                    "relative_path": relative.as_posix(),
                    "split": split,
                    "shot": shot,
                    "crop": "Apple",
                    "category": category,
                    "binary_label": str(label),
                    "source_image_id": original["image_id"],
                    "sha256": actual_hash,
                    "size_bytes": str(path.stat().st_size),
                    "available_in_repository": "false",
                }
            )
    return rows


def write_metadata(root: Path, dataset: str, rows: list[dict[str, str]]) -> dict[str, object]:
    target = root / "metadata" / dataset
    target.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: row["relative_path"])
    with (target / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(
        (row["split"], row["shot"], row["crop"], row["category"], row["binary_label"])
        for row in rows
    )
    count_fields = ["split", "shot", "crop", "category", "binary_label", "count"]
    with (target / "class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=count_fields)
        writer.writeheader()
        for key, count in sorted(counts.items()):
            writer.writerow(dict(zip(count_fields, (*key, count))))

    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    duplicate_groups = {
        digest: members for digest, members in by_hash.items() if len(members) > 1
    }
    duplicate_fields = [
        "sha256",
        "occurrences",
        "splits",
        "shots",
        "binary_labels",
        "relative_paths",
    ]
    with (target / "duplicate_groups.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=duplicate_fields)
        writer.writeheader()
        for digest, members in sorted(duplicate_groups.items()):
            writer.writerow(
                {
                    "sha256": digest,
                    "occurrences": len(members),
                    "splits": ";".join(sorted({row["split"] for row in members})),
                    "shots": ";".join(
                        sorted({row["shot"] for row in members if row["shot"]}, key=int)
                    ),
                    "binary_labels": ";".join(
                        sorted({row["binary_label"] for row in members})
                    ),
                    "relative_paths": ";".join(
                        row["relative_path"] for row in members
                    ),
                }
            )

    train_hashes = {row["sha256"] for row in rows if row["split"] == "train"}
    test_hashes = {row["sha256"] for row in rows if row["split"] == "test"}
    cross_shot_overlap_hashes = sum(
        len({row["shot"] for row in members if row["split"] == "train"}) > 1
        for members in by_hash.values()
    )
    conflicting_label_hashes = sum(
        len({row["binary_label"] for row in members}) > 1
        for members in by_hash.values()
    )

    train_counts = Counter(row["shot"] for row in rows if row["split"] == "train")
    test_normal = sum(
        row["split"] == "test" and row["binary_label"] == "0" for row in rows
    )
    test_anomaly = sum(
        row["split"] == "test" and row["binary_label"] == "1" for row in rows
    )
    summary = {
        "dataset": dataset,
        "total_selected_images": len(rows),
        "training_counts_by_shot": {key: train_counts[key] for key in sorted(train_counts, key=int)},
        "test_normal": test_normal,
        "test_anomaly": test_anomaly,
        "test_total": test_normal + test_anomaly,
        "images_in_repository": rows[0]["available_in_repository"] == "true",
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "manifest_hash_algorithm": "SHA-256",
        "integrity": {
            "unique_sha256": len(by_hash),
            "duplicate_hash_groups": len(duplicate_groups),
            "duplicate_rows_beyond_first": sum(
                len(members) - 1 for members in duplicate_groups.values()
            ),
            "cross_shot_overlap_hashes": cross_shot_overlap_hashes,
            "train_test_overlap_hashes": len(train_hashes & test_hashes),
            "conflicting_binary_label_hashes": conflicting_label_hashes,
        },
    }
    with (target / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def sanitize_audits(args: argparse.Namespace, root: Path) -> None:
    for source_file, target_file in (
        (
            args.plant_village / "manifests/split_summary.json",
            root / "metadata/PlantVillage/split_audit.json",
        ),
        (
            args.plant_pathology / "metadata/split_summary.json",
            root / "metadata/PlantPathology/split_audit.json",
        ),
        (
            args.plant_pathology / "metadata/integrity_report.json",
            root / "metadata/PlantPathology/integrity_report.json",
        ),
    ):
        payload = json.loads(source_file.read_text(encoding="utf-8"))
        for key in (
            "source",
            "target",
            "created_at_unix",
            "supplemented_at_unix",
            "test_good_supplement_source",
            "transfer_modes",
        ):
            payload.pop(key, None)
        target_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    source = args.plant_city / "metadata/label_conflict_resolution.csv"
    target = root / "metadata/PlantCity/label_conflict_resolution.csv"
    with source.open(newline="", encoding="utf-8") as in_handle, target.open(
        "w", newline="", encoding="utf-8"
    ) as out_handle:
        fields = [
            "sha256",
            "removed_relative_path",
            "retained_relative_path",
            "resolution",
            "reason",
        ]
        writer = csv.DictWriter(out_handle, fieldnames=fields)
        writer.writeheader()
        for row in csv.DictReader(in_handle):
            def relative(value: str) -> str:
                marker = "/PlantCity_limit/"
                return value.split(marker, 1)[-1]
            writer.writerow(
                {
                    "sha256": row["sha256"],
                    "removed_relative_path": relative(row["removed_from_active_test"]),
                    "retained_relative_path": relative(row["retained_in_active_test"]),
                    "resolution": row["resolution"],
                    "reason": row["reason"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-village", required=True, type=Path)
    parser.add_argument("--plant-pathology", required=True, type=Path)
    parser.add_argument("--plant-city", required=True, type=Path)
    parser.add_argument("--apple-leaf", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    sources = {
        "PlantVillage": args.plant_village,
        "AppleLeaf9": args.apple_leaf,
        "PlantCity": args.plant_city,
    }
    summaries = []
    for dataset, source in sources.items():
        rows = copy_and_manifest(dataset, source, args.output / "data" / dataset)
        summaries.append(write_metadata(args.output, dataset, rows))

    pp_rows = plant_pathology_manifest(args.plant_pathology)
    summaries.append(write_metadata(args.output, "PlantPathology", pp_rows))
    sanitize_audits(args, args.output)

    summary_fields = [
        "dataset",
        "training_2",
        "training_4",
        "training_8",
        "training_16",
        "test_normal",
        "test_anomaly",
        "test_total",
        "images_in_repository",
        "total_selected_images",
        "total_bytes",
        "duplicate_hash_groups",
        "cross_shot_overlap_hashes",
        "train_test_overlap_hashes",
        "conflicting_binary_label_hashes",
    ]
    with (args.output / "metadata/dataset_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "dataset": summary["dataset"],
                    "training_2": summary["training_counts_by_shot"].get("2", 0),
                    "training_4": summary["training_counts_by_shot"].get("4", 0),
                    "training_8": summary["training_counts_by_shot"].get("8", 0),
                    "training_16": summary["training_counts_by_shot"].get("16", 0),
                    "test_normal": summary["test_normal"],
                    "test_anomaly": summary["test_anomaly"],
                    "test_total": summary["test_total"],
                    "images_in_repository": summary["images_in_repository"],
                    "total_selected_images": summary["total_selected_images"],
                    "total_bytes": summary["total_bytes"],
                    "duplicate_hash_groups": summary["integrity"]["duplicate_hash_groups"],
                    "cross_shot_overlap_hashes": summary["integrity"]["cross_shot_overlap_hashes"],
                    "train_test_overlap_hashes": summary["integrity"]["train_test_overlap_hashes"],
                    "conflicting_binary_label_hashes": summary["integrity"]["conflicting_binary_label_hashes"],
                }
            )
    print("Release data and metadata built successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
