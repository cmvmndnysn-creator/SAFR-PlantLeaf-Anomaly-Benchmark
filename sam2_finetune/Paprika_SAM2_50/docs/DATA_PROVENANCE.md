# Data provenance and processing record

## Export record

- Roboflow project: `Paprika_SAM2`
- Dataset version: `v2`
- Export date: 2026-10-11
- Export format: COCO Segmentation
- Exported images: 50
- Annotation class: `leaf`
- COCO image records: 50
- COCO annotation records: 51

## Distribution authorization

The original data provider authorized public distribution of this 50-image
subset through the present research repository. The original rightsholder
retains copyright in the source images. See `../DATA_USE_NOTICE.md` for the
scope of this notice; the repository-level MIT license does not apply to these
source images or their annotations.

## Roboflow preprocessing

- Auto-orientation with EXIF orientation removal
- Stretch resize to 432 x 432 pixels
- No augmentation

## Mask conversion

COCO polygons were rasterized at the exported image resolution. All annotations
belonging to the same image were merged into one semantic binary mask. Therefore,
`000458_jpg.rf.16c1423cdfb507da406edf705643eeb5.jpg`, which has two leaf
annotations, contains two foreground components in its binary mask.

The canonical masks in `dataset/masks/` use values `0` and `255`. The included
SAM2 conversion script changes these to palette IDs `0` and `1` and creates the
one-frame-per-sample VOS directory structure.

## Integrity and duplicate audit

- Exact file-duplicate groups: 0
- Exact decoded-pixel duplicate groups: 0
- Near-duplicate candidates at dHash Hamming distance <= 8: 0
- Missing image-mask pairs: 0
- Empty masks: 0

The audit above concerns duplicates within this 50-image subset. Overlap with an
external anomaly-detection test split must be checked against that split's files
or manifest. `dataset/splits/mask_training_50.txt`, `manifest.csv`, and SHA-256
checksums make that comparison possible.

