# Generating model weights

A deployable model is a folder `models/<model-id>/` containing two files:

- `metadata.json` — describes the model for the consumer (CLUE).
- `weights.json` — the model's weights in TensorFlow.js JSON format, exported
  from a trained PyTorch checkpoint.

This guide covers producing those files. `compact-v1` was created this way (see
[models/compact-v1/README.md](../models/compact-v1/README.md) for its provenance).

## 1. Train a model

Run the training notebook to produce a PyTorch checkpoint:

- `notebooks/03_training/train_cnn_multiclass.ipynb` (compact architecture)

This writes a checkpoint such as `models/seismic_cnn_compact_<...>.pth`.

## 2. Export to TensorFlow.js JSON

Convert the checkpoint to the TF.js weight format with the export script:

```bash
python scripts/export_compact_weights_for_tfjs.py \
  --model_path models/seismic_cnn_compact_<...>.pth \
  --output_dir /tmp/export
```

This writes `/tmp/export/compact_weights.json` in the same JSON format used by
`weights.json`. The architecture in the script must match the trained model —
`export_compact_weights_for_tfjs.py` targets the compact CNN; add a sibling
exporter for other architectures.

> Note: `scripts/generate_dummy_weights.py` produces *random placeholder* weights
> for the browser benchmark only — do not use it for real deployed models.

## 3. Assemble the model folder

```bash
mkdir -p models/<model-id>
cp /tmp/export/compact_weights.json models/<model-id>/weights.json
# then create models/<model-id>/metadata.json from the template
```

Use the metadata template and field reference in
[docs/deploy-model-for-clue.md](deploy-model-for-clue.md). Bump `id`
(e.g. `compact-v2`) when retraining so events keyed on the old id remain valid.

## 4. Deploy (Concord-only)

Concord deploys the folder to S3 for CLUE with `scripts/deploy-model.sh`; see
[docs/deploy-model-for-clue.md](deploy-model-for-clue.md). This step uses Concord
infrastructure (the `models-resources` S3 bucket) and is specific to Concord's
deployment. Steps 1–3 (train → export → assemble) are general and apply to anyone
producing a model in this format.
