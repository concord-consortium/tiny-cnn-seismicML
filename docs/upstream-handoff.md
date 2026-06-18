# Handoff: updating the lab (upstream) repo

These steps bring `Denolle-Lab/tiny-cnn-seismicML` up to date with Derek's explainer
app and newest training notebooks. Intended for the lab maintainer.

## 1. Add Derek's fork as a remote

```bash
git remote add derek https://github.com/yaoderek/tiny-cnn-seismicML.git
git fetch derek
```

## 2. Bring in the explainer app and export scripts (Derek's version)

Take the `explainer-app/` directory and the two export scripts from `derek/main`:

```bash
git checkout derek/main -- \
  explainer-app \
  scripts/export_waveforms_for_explainer.py \
  scripts/export_compact_weights_for_tfjs.py
```

`export_compact_weights_for_tfjs.py` converts a trained PyTorch checkpoint to the
TensorFlow.js `weights.json` format; it pairs with the model-weights guide in step 4.

## 3. Update training notebooks to Derek's newer versions

```bash
git checkout derek/main -- \
  notebooks/02_labeling/download_AK_only_data.ipynb \
  notebooks/03_training/train_cnn_multiclass.ipynb
```

Confirm with Derek that these are the intended current versions before merging.

## 4. Model-weights pattern (contributed separately by Concord)

So the lab can produce its own deployable models, Concord has opened
[PR #6](https://github.com/Denolle-Lab/tiny-cnn-seismicML/pull/6) adding the
model-weights pattern:

- `docs/generating-model-weights.md` — the train → export → assemble workflow (uses
  `export_compact_weights_for_tfjs.py`, which arrives with the explainer app in step 2).
- An example `models/compact-v1/` folder (`metadata.json` + `weights.json`) as a
  template.

The export script itself is Derek's and comes in via step 2 — it is **not** part of
this Concord PR. Concord's S3 deploy tooling (`scripts/deploy-model.sh`,
`docs/deploy-model-for-clue.md`) is also intentionally **not** contributed upstream —
it is Concord infrastructure. Note that `models/compact-v1/metadata.json` references a
CLUE-specific `$schema`; adjust or generalize it if the lab's models are not consumed
by CLUE.
