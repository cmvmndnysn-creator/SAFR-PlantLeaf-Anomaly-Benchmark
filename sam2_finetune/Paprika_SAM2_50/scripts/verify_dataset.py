#!/usr/bin/env python3
"""Verify the public Paprika SAM2 source dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    images_dir = DATASET / "images"
    masks_dir = DATASET / "masks"
    manifest_path = DATASET / "metadata" / "manifest.csv"
    split_path = DATASET / "splits" / "mask_training_50.txt"
    coco_path = DATASET / "annotations" / "_annotations.coco.json"

    images = {path.stem: path for path in images_dir.glob("*.jpg")}
    masks = {path.stem: path for path in masks_dir.glob("*.png")}
    assert len(images) == 50, f"Expected 50 images, found {len(images)}"
    assert len(masks) == 50, f"Expected 50 masks, found {len(masks)}"
    assert images.keys() == masks.keys(), "Image and mask stems do not match"

    with manifest_path.open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    assert len(manifest) == 50, f"Expected 50 manifest rows, found {len(manifest)}"

    manifest_by_stem = {Path(row["source_file"]).stem: row for row in manifest}
    assert manifest_by_stem.keys() == images.keys(), "Manifest identities do not match images"

    for stem in sorted(images):
        image_path = images[stem]
        mask_path = masks[stem]
        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(mask_path) as mask:
            mask_size = mask.size
            mask_mode = mask.mode
            pixels = (
                mask.get_flattened_data()
                if hasattr(mask, "get_flattened_data")
                else mask.getdata()
            )
            values = set(pixels)
        assert image_size == mask_size, f"Size mismatch: {stem}"
        assert mask_mode == "L", f"Mask is not single-channel L mode: {stem}"
        assert values.issubset({0, 255}) and 255 in values, f"Invalid binary mask: {stem}"
        assert sha256_file(image_path) == manifest_by_stem[stem]["source_sha256"], (
            f"Image checksum mismatch: {stem}"
        )

    split_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(split_ids) == 50 and len(set(split_ids)) == 50, "Split must contain 50 unique IDs"
    assert set(split_ids) == images.keys(), "Split identities do not match image stems"

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    assert len(coco.get("images", [])) == 50, "COCO image count is not 50"
    assert len(coco.get("annotations", [])) == 51, "COCO annotation count is not 51"

    print(
        json.dumps(
            {
                "images": len(images),
                "masks": len(masks),
                "manifest_rows": len(manifest),
                "split_identities": len(split_ids),
                "coco_annotations": len(coco["annotations"]),
                "status": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
