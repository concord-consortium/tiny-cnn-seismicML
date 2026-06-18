# Handoff: updating the lab (upstream) repo

These steps bring `Denolle-Lab/tiny-cnn-seismicML` up to date with Derek's explainer
app and newest training notebooks. Intended for the lab maintainer.

## 1. Add Derek's fork as a remote

```bash
git remote add derek https://github.com/yaoderek/tiny-cnn-seismicML.git
git fetch derek
```

## 2. Bring in the explainer app (Derek's version)

Take the `explainer-app/` directory and `scripts/export_waveforms_for_explainer.py`
from `derek/main`:

```bash
git checkout derek/main -- explainer-app scripts/export_waveforms_for_explainer.py
```

Note: Derek's explainer app does **not** include the benchmark. The benchmark files
(`benchmark.html`, `src/benchmark.js`, `benchmark.md`, `apps-script/Code.gs`) and the
`benchmark` rollup input in `vite.config.js` are Concord's additions — they upload to
a Concord-owned Google Sheet and live only in the Concord fork. `derek/main`'s
`explainer-app/` is already benchmark-free, so there is nothing to exclude here.

## 3. Update training notebooks to Derek's newer versions

```bash
git checkout derek/main -- \
  notebooks/02_labeling/download_AK_only_data.ipynb \
  notebooks/03_training/train_cnn_multiclass.ipynb
```

Confirm with Derek that these are the intended current versions before merging.

## 4. Model-weights pattern (contributed separately by Concord)

So the lab can produce its own deployable models, Concord will open a separate PR
adding the model-weights pattern:

- `scripts/export_compact_weights_for_tfjs.py` — exports a trained checkpoint to
  TensorFlow.js JSON `weights.json`.
- `docs/generating-model-weights.md` — the train → export → assemble workflow.
- An example `models/compact-v1/` folder (`metadata.json` + `weights.json`) as a
  template.

Concord's S3 deploy tooling (`scripts/deploy-model.sh`, `docs/deploy-model-for-clue.md`)
is intentionally **not** contributed upstream — it is Concord infrastructure. Note
that `models/compact-v1/metadata.json` references a CLUE-specific `$schema`; adjust
or generalize it if the lab's models are not consumed by CLUE.

## 5. Review and open PR(s)

Commit on a branch and open one or more PRs for review. Concord can assist with the
explainer-app PR if helpful.

## After adoption

Concord will sync by merging `upstream/main` into its fork's `main`; the explainer app
then comes from upstream and no longer appears as a Concord-specific change.
