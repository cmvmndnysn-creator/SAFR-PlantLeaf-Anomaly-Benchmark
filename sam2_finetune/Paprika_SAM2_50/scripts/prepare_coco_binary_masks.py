#!/usr/bin/env python3
"""Audit a COCO segmentation export and build paired binary PNG masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}xRGB".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def phash64(image: Image.Image) -> int:
    """Dependency-free 64-bit difference hash."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(
        gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata()
    )
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(
            pixels[offset + column] > pixels[offset + column + 1]
            for column in range(8)
        )
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def safe_stem(file_name: str) -> str:
    return Path(file_name).stem


def find_annotation_file(train_dir: Path) -> Path:
    preferred = train_dir / "_annotations.coco.json"
    if preferred.exists():
        return preferred
    candidates = sorted(train_dir.glob("*.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one COCO JSON file in {train_dir}, found {len(candidates)}"
        )
    return candidates[0]


def load_font(size: int = 12) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def build_preview(rows: list[dict], output_path: Path) -> None:
    columns = 5
    tile_w, tile_h = 220, 180
    sheet_rows = math.ceil(len(rows) / columns)
    canvas = Image.new("RGB", (columns * tile_w, sheet_rows * tile_h), "white")
    font = load_font(12)

    for index, row in enumerate(rows):
        image = Image.open(row["output_image"]).convert("RGB")
        mask = Image.open(row["output_mask"]).convert("L")
        overlay = image.copy()
        tint = Image.new("RGB", image.size, (255, 60, 60))
        alpha = mask.point(lambda p: 105 if p else 0)
        overlay.paste(tint, (0, 0), alpha)
        overlay.thumbnail((tile_w - 12, tile_h - 34), Image.Resampling.LANCZOS)

        tile = Image.new("RGB", (tile_w, tile_h), "white")
        x = (tile_w - overlay.width) // 2
        y = 4 + (tile_h - 34 - overlay.height) // 2
        tile.paste(overlay, (x, y))
        label = f"{index + 1:02d}  {row['source_file']}"
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((6, tile_h - 25), label, fill="black", font=font)
        canvas.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))

    canvas.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="COCO train directory")
    parser.add_argument("output_dir", type=Path, help="New output directory")
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=8,
        help="Maximum 64-bit pHash Hamming distance to report",
    )
    args = parser.parse_args()

    train_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")

    annotation_path = find_annotation_file(train_dir)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    if not images:
        raise ValueError("COCO file contains no images")

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    reports_dir = output_dir / "reports"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    manifest_rows: list[dict] = []
    audit_images: list[dict] = []
    errors: list[str] = []
    file_hash_groups: dict[str, list[str]] = defaultdict(list)
    pixel_hash_groups: dict[str, list[str]] = defaultdict(list)

    for image_record in sorted(images, key=lambda item: item["file_name"]):
        source_path = train_dir / image_record["file_name"]
        if not source_path.exists():
            errors.append(f"Missing image: {image_record['file_name']}")
            continue

        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        expected_width = int(image_record.get("width", width))
        expected_height = int(image_record.get("height", height))
        if (width, height) != (expected_width, expected_height):
            errors.append(
                f"Size mismatch for {source_path.name}: file={width}x{height}, "
                f"COCO={expected_width}x{expected_height}"
            )

        image_annotations = annotations_by_image.get(image_record["id"], [])
        if not image_annotations:
            errors.append(f"No annotations: {source_path.name}")

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        polygon_count = 0
        for annotation in image_annotations:
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list):
                errors.append(
                    f"Unsupported non-polygon segmentation in {source_path.name}, "
                    f"annotation {annotation.get('id')}"
                )
                continue
            for polygon in segmentation:
                if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                    errors.append(
                        f"Invalid polygon in {source_path.name}, annotation {annotation.get('id')}"
                    )
                    continue
                points = [
                    (
                        min(max(float(polygon[index]), 0.0), width - 1.0),
                        min(max(float(polygon[index + 1]), 0.0), height - 1.0),
                    )
                    for index in range(0, len(polygon), 2)
                ]
                draw.polygon(points, fill=255)
                polygon_count += 1

        foreground_pixels = int(mask.histogram()[255])
        if foreground_pixels == 0:
            errors.append(f"Empty mask: {source_path.name}")

        copied_image = images_dir / source_path.name
        output_mask = masks_dir / f"{safe_stem(source_path.name)}.png"
        shutil.copy2(source_path, copied_image)
        mask.save(output_mask, optimize=True)

        file_hash = sha256_file(source_path)
        pixel_hash = decoded_pixel_hash(image)
        file_hash_groups[file_hash].append(source_path.name)
        pixel_hash_groups[pixel_hash].append(source_path.name)
        perceptual_hash = phash64(image)

        class_names = sorted(
            {
                categories.get(annotation.get("category_id"), str(annotation.get("category_id")))
                for annotation in image_annotations
            }
        )
        manifest_rows.append(
            {
                "source_file": source_path.name,
                "mask_file": output_mask.name,
                "coco_image_id": image_record["id"],
                "width": width,
                "height": height,
                "annotation_count": len(image_annotations),
                "polygon_count": polygon_count,
                "classes": ";".join(class_names),
                "foreground_pixels": foreground_pixels,
                "foreground_fraction": f"{foreground_pixels / (width * height):.8f}",
                "source_sha256": file_hash,
            }
        )
        audit_images.append(
            {
                "source_file": source_path.name,
                "file_hash": file_hash,
                "pixel_hash": pixel_hash,
                "phash": perceptual_hash,
                "output_image": copied_image,
                "output_mask": output_mask,
            }
        )

    near_duplicate_rows: list[dict] = []
    for left_index, left in enumerate(audit_images):
        for right in audit_images[left_index + 1 :]:
            distance = hamming64(left["phash"], right["phash"])
            if distance <= args.near_duplicate_threshold:
                near_duplicate_rows.append(
                    {
                        "file_a": left["source_file"],
                        "file_b": right["source_file"],
                        "phash_hamming_64": distance,
                        "classification": "strong_candidate" if distance <= 4 else "review",
                    }
                )
    near_duplicate_rows.sort(key=lambda row: (row["phash_hamming_64"], row["file_a"], row["file_b"]))

    exact_file_groups = [names for names in file_hash_groups.values() if len(names) > 1]
    exact_pixel_groups = [names for names in pixel_hash_groups.values() if len(names) > 1]

    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (reports_dir / "near_duplicate_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = ["file_a", "file_b", "phash_hamming_64", "classification"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(near_duplicate_rows)

    duplicate_report = {
        "exact_file_duplicate_groups": exact_file_groups,
        "exact_decoded_pixel_duplicate_groups": exact_pixel_groups,
        "near_duplicate_threshold_phash_hamming_64": args.near_duplicate_threshold,
        "near_duplicate_candidates": near_duplicate_rows,
    }
    (reports_dir / "duplicate_report.json").write_text(
        json.dumps(duplicate_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "source_directory": str(train_dir),
        "source_annotation_file": str(annotation_path),
        "image_count_in_coco": len(images),
        "annotation_count_in_coco": len(annotations),
        "generated_mask_count": len(manifest_rows),
        "category_names": sorted(set(categories.values())),
        "images_with_multiple_annotations": [
            row["source_file"] for row in manifest_rows if row["annotation_count"] > 1
        ],
        "exact_file_duplicate_group_count": len(exact_file_groups),
        "exact_decoded_pixel_duplicate_group_count": len(exact_pixel_groups),
        "near_duplicate_candidate_count": len(near_duplicate_rows),
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    build_preview(audit_images, output_dir / "mask_overlay_preview.jpg")

    readme = f"""# Paprika SAM2 binary-mask dataset

- Source: `{train_dir}`
- Images: {len(manifest_rows)}
- COCO annotations: {len(annotations)}
- Mask format: single-channel PNG, background `0`, leaf foreground `255`
- Mask dimensions: identical to the paired image
- Multiple polygons/instances belonging to one image are merged into one binary mask

Files are paired by stem between `images/` and `masks/`. See `manifest.csv` for the
complete identity list and `reports/duplicate_report.json` for the duplicate audit.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
