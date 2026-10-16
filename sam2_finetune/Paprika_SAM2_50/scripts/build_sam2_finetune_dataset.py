#!/usr/bin/env python3
"""Convert paired images and 0/255 masks to the SAM2 PNGRawDataset layout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paired_dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--class-name", default="leaf")
    args = parser.parse_args()

    source = args.paired_dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output}")

    source_images = source / "images"
    source_masks = source / "masks"
    image_paths = sorted(
        path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No images found in {source_images}")

    training_data = output / "training_data" / args.class_name
    training_masks = output / "training_masks" / args.class_name
    vos_images = output / "vos_images"
    vos_masks = output / "vos_masks"
    provenance = output / "provenance"
    for directory in (training_data, training_masks, vos_images, vos_masks, provenance):
        directory.mkdir(parents=True, exist_ok=True)

    sample_ids: list[str] = []
    manifest_rows: list[dict] = []

    for image_path in image_paths:
        mask_path = source_masks / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        with Image.open(image_path) as opened_image:
            image_size = opened_image.size
        with Image.open(mask_path) as opened_mask:
            binary_mask = opened_mask.convert("L")
        if binary_mask.size != image_size:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: "
                f"{image_size} vs {binary_mask.size}"
            )

        pixels = (
            binary_mask.get_flattened_data()
            if hasattr(binary_mask, "get_flattened_data")
            else binary_mask.getdata()
        )
        values = set(pixels)
        if not values.issubset({0, 255}) or 255 not in values:
            raise ValueError(
                f"Mask must be non-empty and binary 0/255: {mask_path.name}, "
                f"values={sorted(values)}"
            )

        sample_id = f"{args.class_name}_{image_path.stem}"
        sample_ids.append(sample_id)

        copied_training_image = training_data / image_path.name
        copied_training_mask = training_masks / mask_path.name
        shutil.copy2(image_path, copied_training_image)
        shutil.copy2(mask_path, copied_training_mask)

        sample_image_dir = vos_images / sample_id
        sample_mask_dir = vos_masks / sample_id
        sample_image_dir.mkdir()
        sample_mask_dir.mkdir()
        shutil.copy2(image_path, sample_image_dir / "00000.jpg")

        palette_mask = binary_mask.point(lambda value: 1 if value > 128 else 0, mode="P")
        palette = [0] * 768
        palette[3:6] = [128, 0, 0]
        palette_mask.putpalette(palette)
        palette_mask.save(sample_mask_dir / "00000.png", optimize=True)

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source_image": image_path.name,
                "source_mask": mask_path.name,
                "width": image_size[0],
                "height": image_size[1],
                "vos_image": f"vos_images/{sample_id}/00000.jpg",
                "vos_mask": f"vos_masks/{sample_id}/00000.png",
            }
        )

    (output / "file_list.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    for name in ("manifest.csv", "summary.json", "mask_overlay_preview.jpg", "README.md"):
        source_item = source / name
        if source_item.exists():
            shutil.copy2(source_item, provenance / name)
    source_reports = source / "reports"
    if source_reports.exists():
        shutil.copytree(source_reports, provenance / "reports")

    source_coco_candidates = [
        source / "annotations" / "_annotations.coco.json",
        source / "_annotations.coco.json",
        source / "provenance" / "_annotations.coco.json",
    ]
    for source_coco in source_coco_candidates:
        if source_coco.exists():
            shutil.copy2(source_coco, provenance / "_annotations.coco.json")
            break
    source_readme_candidates = [
        source / "README.roboflow.txt",
        source / "provenance" / "README.roboflow.txt",
    ]
    for source_readme in source_readme_candidates:
        if source_readme.exists():
            shutil.copy2(source_readme, provenance / source_readme.name)
            break

    override = f"""dataset:
  img_folder: {output / 'vos_images'}
  gt_folder: {output / 'vos_masks'}
  file_list_txt: {output / 'file_list.txt'}
"""
    (output / "sam2_dataset_override.yaml").write_text(override, encoding="utf-8")

    readme = f"""# Paprika SAM2 fine-tuning dataset

This directory matches the SAM2 `PNGRawDataset` layout used by the existing
server training configuration.

- Samples: {len(sample_ids)}
- Class: `{args.class_name}`
- `vos_images/<sample_id>/00000.jpg`: one static-image frame per sample
- `vos_masks/<sample_id>/00000.png`: palette PNG with background ID `0` and
  leaf ID `1`
- `file_list.txt`: exact sample identities used by the training loader
- `training_data/` and `training_masks/`: pre-conversion 0/255 image-mask pairs
- `provenance/`: COCO source annotations, duplicate audit, and mask preview
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "sample_count": len(sample_ids),
        "class_name": args.class_name,
        "image_directory_count": len(list(vos_images.iterdir())),
        "mask_directory_count": len(list(vos_masks.iterdir())),
        "file_list_count": len(sample_ids),
        "mask_format": "palette PNG",
        "mask_ids": [0, 1],
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
