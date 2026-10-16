# SAFR Plant Leaf Anomaly Detection Benchmark

This repository contains the processed benchmark partitions used for few-shot
plant leaf anomaly detection in *Few-Shot Plant Leaf Anomaly Detection via
Structure-Aware Feature Reconstruction*.

The benchmark reorganizes selected images from four public datasets. Healthy
leaves are normal samples and diseased leaves are anomalous samples. The
few-shot training sets contain healthy images only.

The repository also includes a separately organized 50-image paprika leaf
segmentation set with manually reviewed masks for SAM2 fine-tuning. It is kept
outside the four anomaly-detection benchmark partitions; see
[`sam2_finetune/Paprika_SAM2_50`](sam2_finetune/Paprika_SAM2_50/).

## Benchmark overview

| Dataset | Crops | Training images (2/4/8/16-shot) | Test normal | Test anomaly | Test total | Images in this repository |
|---|---:|---:|---:|---:|---:|---|
| PlantVillage | 9 | 18 / 36 / 72 / 144 | 270 | 720 | 990 | Yes |
| AppleLeaf9 | 1 | 2 / 4 / 8 / 16 | 80 | 240 | 320 | Yes |
| Plant Pathology 2021 | 1 | 2 / 4 / 8 / 16 | 462 | 462 | 924 | **No; manifest only** |
| PlantCity | 11 | 22 / 44 / 88 / 176 | 2,278 | 4,120 | 6,398 | Yes |

The four support-size partitions are deterministic and are not nested subsets.
PlantVillage, Plant Pathology, and PlantCity use mutually disjoint support
partitions. AppleLeaf9 retains five same-label healthy images that occur in more
than one shot setting, matching the experiments as run. Images used for testing
are excluded from every support set. See `metadata/<dataset>/manifest.csv` for
the identity and SHA-256 digest of every selected image.

The integrity audit also records 35 same-label byte-identical groups within the
PlantCity test set. There are no normal/anomaly hash conflicts and no train/test
hash overlaps in any dataset. These duplicates are disclosed rather than
silently removed because this release is intended to reproduce the reported
partitions and results. See each `duplicate_groups.csv` and `summary.json`.

## Why Plant Pathology images are not included

The Plant Pathology 2021 competition rules prohibit redistribution of the
competition data to people who have not accepted the Kaggle rules. Consequently,
this repository publishes only the selected image identifiers, labels, hashes,
and split paths. After downloading the official data from Kaggle, reconstruct
the local subset with:

```bash
python3 scripts/materialize_plant_pathology.py \
  --source /path/to/plant-pathology-2021-fgvc8 \
  --output data/PlantPathology
```

The generated Plant Pathology image folders are ignored by Git and must not be
committed.

## Repository layout

```text
.
├── data/
│   ├── PlantVillage/       # selected images
│   ├── AppleLeaf9/         # selected images
│   ├── PlantCity/          # selected images
│   └── PlantPathology/     # README only; reconstructed locally
├── metadata/
│   ├── dataset_summary.csv
│   └── <dataset>/
│       ├── manifest.csv
│       ├── class_counts.csv
│       └── summary.json
├── scripts/
│   ├── materialize_plant_pathology.py
│   └── verify_dataset.py
├── sam2_finetune/
│   └── Paprika_SAM2_50/    # 50 images and masks for SAM2 fine-tuning
├── DATASET_CARD.md
├── THIRD_PARTY_NOTICES.md
└── CITATION.cff
```

Within a dataset, `good` denotes a healthy/normal image. Other test categories
denote anomalies. Training directory names encode the support size, for example
`train_8` contains the 8-shot selection for each retained crop.

## SAM2 mask fine-tuning subset

`sam2_finetune/Paprika_SAM2_50` contains 50 paprika leaf images, their
single-channel binary PNG masks, the source COCO segmentation annotations, and
the exact image-identity list in `dataset/splits/mask_training_50.txt`. The
directory also includes image and mask checksums, duplicate checks, and scripts
to generate the SAM2 `PNGRawDataset` layout. This subset is documented
separately because it is a segmentation fine-tuning resource, rather than one
of the four anomaly-detection benchmark partitions. See its `README.md` and
`docs/DATA_PROVENANCE.md` for details and `DATA_USE_NOTICE.md` for its
source-image distribution notice.

## Integrity verification

Run the verifier after cloning and downloading Git LFS objects:

```bash
python3 scripts/verify_dataset.py
```

To require a locally reconstructed Plant Pathology subset as well:

```bash
python3 scripts/verify_dataset.py --require-plant-pathology
```

## Large-file storage

The image files are configured for Git LFS in `.gitattributes`. Install Git LFS
before the first `git add` or after cloning this repository.

## Sources and licenses

This is a derived benchmark, not a new claim of ownership over the source
images. The root code license does not replace the source dataset licenses.

- **PlantVillage:** original RGB/color images without augmentation, corresponding
  to the non-augmented PlantVillage release. CC0 1.0 release:
  <https://doi.org/10.17632/tywbtsjrjv.1>. Original repository:
  <https://github.com/spMohanty/PlantVillage-Dataset>.
- **AppleLeaf9:** CC BY 4.0,
  <https://github.com/JasonYangCode/AppleLeaf9>.
- **PlantCity:** CC BY 4.0, version 1,
  <https://doi.org/10.17632/w8kh2xkspx.1>.
- **Plant Pathology 2021:** competition and academic research use subject to the
  Kaggle competition rules,
  <https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8>.
- **Paprika SAM2 fine-tuning subset:** public distribution of the 50-image
  subset is authorized by its original data provider; copyright in the source
  images remains with the original rightsholder. See
  `sam2_finetune/Paprika_SAM2_50/DATA_USE_NOTICE.md`.

See `THIRD_PARTY_NOTICES.md` and the per-dataset README files for attribution and
citation details.

## Citation

Use the repository's `CITATION.cff` metadata and cite the original dataset
publications listed in `THIRD_PARTY_NOTICES.md`.
