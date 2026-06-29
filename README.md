# DTMf-MIL

Code release for the DTMf-MIL / `spatial_abmil_orig`.

- `spatial_abmil_orig`, the original DTMf-MIL in code `spatial_abmil_orig` model.
- Standard MIL baselines: ABMIL, CLAM, DSMIL, DFTD, ILRA, RRT-MIL, TransMIL, Transformer MIL, and WIKG.
- SDF/RBF feature extraction utilities.
- A generalized training script that can train `spatial_abmil_orig` and the baselines on any fold configuration (including single splits).

## Repository Layout

```text
DTMf-MIL/
  train.py                       # Training script
  WSI_dataset.py                 # MIL dataset loader with visual features + SDF features
  preprocessing/
    cv_kmeans.py                 # KMeans fitting
    cv_extract_sdf.py            # SDF/RBF extraction
  src/
    builder.py                   # model builder
    _global_mappings.py          # model registry
    models/
      spatial_abmil_orig.py      # DTMf-MIL model
      *.py                       # MIL models
    model_configs/               # YAML configs for included models only
  splits/                        # Dataset split jsons
```

## Data Expectations

`train.py` expects each backbone folder to contain visual features and fold-specific SDF features. For a single split setup (1-fold), the folder structure is:

```text
BASE_ROOT/
  BACKBONE_NAME/
    training/
      CLASS_NAME/
        pt_files/*.pt
        h5_files/*.h5
    testing/
      pt_files/*.pt
    # Fold-specific SDF folder (e.g. fold_1)
    sdf_70d_features_fold_1/*{SDF_SUFFIX}.npz
```

Each SDF file should contain:

- `spatial_features`: `[num_patches, 70]`
- `coords`: `[num_patches, 2]`
- `labels`: `[num_patches]`

The 70 spatial channels are `5` KMeans phenotype clusters x `14` features per cluster:

- 10 RBF channels from the signed distance field
- SDF gradient x
- SDF gradient y
- local mean SDF
- local std SDF

---

## Split JSON Structure

To run a single train/val/test split, create a split JSON with a single fold key (e.g. `"fold_1"`):

```json
{
    "fold_1": {
        "train": ["slide_id_1", "slide_id_2", ...],
        "val": ["slide_id_3", "slide_id_4", ...],
        "test": ["slide_id_5", "slide_id_6", ...]
    }
}
```

For N-fold cross-validation, simply add more keys (e.g., `"fold_2"`, `"fold_3"`, ..., `"fold_N"`).

---

## Preprocessing (Single Split Example)

### 1. Fit KMeans
Fit KMeans using only the training slides of the split:

```bash
python preprocessing/cv_kmeans.py \
  --root_dir /path/to/BASE_ROOT/BACKBONE_NAME \
  --out_dir /path/to/BASE_ROOT/BACKBONE_NAME \
  --cv_json /path/to/single_split.json \
  --model_name BACKBONE_NAME \
  --clusters 5
```

### 2. Extract SDF Features
Compute and save the 70D SDF features based on the fitted KMeans. You can customize the file suffix using `--sdf_suffix` (defaults to `_sdf70`):

```bash
python preprocessing/cv_extract_sdf.py \
  --root_dir /path/to/BASE_ROOT/BACKBONE_NAME \
  --model_dir /path/to/BASE_ROOT/BACKBONE_NAME \
  --cv_json /path/to/single_split.json \
  --model_name BACKBONE_NAME \
  --sdf_suffix _sdf70
```

---

## Training (Single Split Example)

To train models on the single split, run `train.py` pointing to the single-split JSON file. Customize the suffix of the loaded SDF features using `--sdf-suffix` (defaults to `_sdf70`):

```bash
python train.py \
  --base-root /path/to/BASE_ROOT \
  --dataset-name hancock \
  --backbones uni \
  --cv-split-json splits/single_split.json \
  --label-json /path/to/test_labels.json \
  --class-map '{"non_metastatic": 0, "metastatic": 1}' \
  --save-dir runs/hancock \
  --model-families spatial_abmil_orig abmil rrt \
  --sdf-suffix _sdf70
```

For binary tasks, checkpoints are selected by validation AUC, with validation loss used only as a tie-breaker. For multi-class tasks, checkpoints are selected by validation QWK, again with validation loss as a tie-breaker.

---

## Included Model Families

Valid `--model-families` entries are:

```text
abmil
clam
dftd
ilra
rrt
transmil
spatial_abmil_orig
```

`spatial_abmil_orig` is the DTMf-MIL model.


The code is adapted from https://github.com/mahmoodlab/MIL-Lab/tree/main. Thanking the authors of this repository for their open source implementation.
