# Plant Pathology 2021 subset

Image files are intentionally not distributed here. Kaggle's Plant Pathology
2021 competition rules prohibit redistribution of competition data to users who
have not accepted those rules.

1. Join the competition or otherwise obtain authorized access at
   <https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8>.
2. Download and extract the official images.
3. From the repository root, run:

```bash
python3 scripts/materialize_plant_pathology.py \
  --source /path/to/extracted/official/data \
  --output data/PlantPathology
```

The selection manifest is `metadata/PlantPathology/manifest.csv`. It contains
954 selected image IDs and SHA-256 digests: 30 mutually disjoint healthy support
images and 924 evaluation images.

