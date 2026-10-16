# Dataset Card

## Dataset summary

This benchmark supports few-shot, normal-only plant leaf anomaly detection.
It contains fixed processed subsets of PlantVillage, AppleLeaf9, Plant
Pathology 2021, and PlantCity. A model is trained using a small healthy support
set and evaluated on healthy and diseased leaves.

## Intended use

- Reproduce the dataset partitions reported in the accompanying manuscript.
- Compare normal-only anomaly-detection methods under fixed 2-, 4-, 8-, and
  16-shot settings.
- Audit train/test membership through file identities and SHA-256 hashes.

The benchmark is not intended to provide medical, agronomic, or pesticide-use
advice. Predictions should not replace expert crop-disease diagnosis.

## Composition

The exact counts are reported in `metadata/dataset_summary.csv`. Three source
subsets are distributed as images in this repository. Plant Pathology 2021 is
metadata-only because its Kaggle terms restrict redistribution.

Each per-image manifest records:

- dataset name;
- relative target path;
- split and shot setting;
- crop and category;
- binary normal/anomaly label;
- source image identifier;
- SHA-256 digest and byte size; and
- whether the image is distributed in the repository.

## Split construction

- Only healthy images occur in support/training partitions.
- Test partitions contain healthy and diseased images.
- Sampling within each shot setting was performed without replacement.
- PlantVillage, Plant Pathology, and PlantCity have mutually disjoint shot
  partitions. AppleLeaf9 contains five same-label healthy hashes reused across
  shot settings, matching the experiments as run.
- File hashes were used to remove byte-identical train/test overlap and resolve
  recorded label conflicts.
- PlantVillage uses deterministic sampling seed 0.
- Plant Pathology 2021 uses seed 20210724.
- PlantCity anomaly subsampling uses seed 20260611.

## Preprocessing

This release preserves the selected image bytes; it does not recompress or
resize the images. Directory reorganization and metadata sanitization do not
alter image content. Absolute paths from the experiment server are intentionally
excluded from the public metadata.

## Known limitations

- The benchmark is a selected subset, not a complete mirror of any source
  dataset.
- Source labels may contain acquisition, annotation, or class-imbalance biases.
- PlantVillage has controlled backgrounds and may overestimate performance in
  field conditions.
- The spelling `Lokat` is retained in PlantCity paths for compatibility with the
  processed benchmark; it refers to loquat.
- Hash-based checking detects identical files, but not every visually related
  acquisition or transformed near-duplicate.
- PlantCity contains 35 same-label byte-identical groups within its test set.
  They are retained for exact experimental reproducibility. No dataset contains
  a train/test hash overlap or a normal/anomaly hash conflict.
- Plant Pathology images require separate authorized access from Kaggle.

## Licensing and attribution

The images retain their original dataset licenses and terms. See
`THIRD_PARTY_NOTICES.md`. The MIT license in this repository applies only to the
new scripts and documentation, not to third-party images.
