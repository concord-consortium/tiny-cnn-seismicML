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
from `derek/main`.

**Exclude the benchmark** — these files point at a Concord-owned Google Sheet and
should not live in the lab repo:

- `explainer-app/benchmark.html`
- `explainer-app/src/benchmark.js`
- `explainer-app/benchmark.md`
- `explainer-app/apps-script/Code.gs`

```bash
git checkout derek/main -- explainer-app scripts/export_waveforms_for_explainer.py
git rm -r --cached explainer-app/apps-script explainer-app/benchmark.html \
  explainer-app/benchmark.md explainer-app/src/benchmark.js 2>/dev/null || true
rm -rf explainer-app/apps-script explainer-app/benchmark.html \
  explainer-app/benchmark.md explainer-app/src/benchmark.js
```

(Also drop the `benchmark` rollup input from `explainer-app/vite.config.js` if present.)

## 3. Update training notebooks to Derek's newer versions

```bash
git checkout derek/main -- \
  notebooks/02_labeling/download_AK_only_data.ipynb \
  notebooks/03_training/train_cnn_multiclass.ipynb
```

Confirm with Derek that these are the intended current versions before merging.

## 4. Review and open PR(s)

Commit on a branch and open one or more PRs for review. Concord can assist with the
explainer-app PR if helpful.

## After adoption

Concord will sync by merging `upstream/main` into its fork's `main`; the explainer app
then comes from upstream and no longer appears as a Concord-specific change.
