# Paprika Leaf Segmentation Masks for SAM2 Fine-Tuning

This repository contains 50 paprika-leaf images and manually reviewed leaf
segmentation annotations. It is organized as a compact, auditable source dataset;
the training-ready SAM2 `PNGRawDataset` layout can be generated with the included
script without storing a second copy of every image and mask in Git.

## Contents

```text
dataset/
  images/                         50 JPEG images
  masks/                          50 binary PNG masks
  annotations/_annotations.coco.json
  splits/mask_training_50.txt     exact identities of the 50 mask-training images
  metadata/manifest.csv           dimensions, annotation counts, and image hashes
  metadata/duplicate_report.json
  metadata/near_duplicate_candidates.csv
  metadata/CHECKSUMS.sha256
docs/
  DATA_PROVENANCE.md
  mask_overlay_preview.jpg
scripts/
  prepare_coco_binary_masks.py
  build_sam2_finetune_dataset.py
  verify_dataset.py
```

## Data format

- Images are 432 x 432 pixels after Roboflow auto-orientation and stretch resize.
- Each image has a same-stem, single-channel PNG mask.
- Binary-mask pixel values are `0` for background and `255` for leaf foreground.
- The original polygon annotations are retained in COCO Segmentation JSON format.
- No augmentation was applied during the Roboflow export.

## Verify the release

```bash
python3 -m pip install -r requirements.txt
python3 scripts/verify_dataset.py
```

## Build the SAM2 fine-tuning layout

```bash
python3 scripts/build_sam2_finetune_dataset.py \
  dataset \
  sam2_ready \
  --class-name leaf
```

The generated directory contains:

```text
vos_images/<sample_id>/00000.jpg
vos_masks/<sample_id>/00000.png
file_list.txt
```

The generated VOS masks are palette PNG files with object IDs `0` (background)
and `1` (leaf), matching the SAM2 `PNGRawDataset` input convention used by the
project training configuration.

## Duplicate audit

The 50 images were checked using file SHA-256 hashes, decoded-pixel hashes, and a
64-bit perceptual difference hash. No exact or near-duplicate candidates were
found within this 50-image subset. See `dataset/metadata/duplicate_report.json`.

## License

The original data provider has authorized public distribution of this 50-image
subset as a research reproducibility resource. Copyright in the source images
remains with the original rightsholder. See `DATA_USE_NOTICE.md` for the scope
of this authorization. The repository-level MIT license applies only to the
repository's original scripts and documentation; it does not relicense the
source images or annotations.

