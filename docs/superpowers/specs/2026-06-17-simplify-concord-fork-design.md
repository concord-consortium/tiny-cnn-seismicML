# Design: Simplify the Concord fork of tiny-cnn-seismicML

**Date:** 2026-06-17
**Status:** Draft (uncommitted — pending fork cleanup)

## Problem

The Concord fork (`origin` = `concord-consortium/tiny-cnn-seismicML`) has become hard to
reason about. A developer cloning it cannot tell what Concord actually added or why the
fork exists, for three reasons:

1. **The fork's `main` is a pure mirror of the lab repo** — all of Concord's real work
   lives on un-merged PR branches (#2, #3), so `main` looks identical to upstream.
2. **The diff is dominated by the explainer app** (~35 files / +7,379 lines ≈ 83% of the
   fork's changes), which is *Derek's* code, not Concord's — and the lab maintainer wants
   to bring it into the lab repo anyway.
3. **Concord-specific and shared concerns are entangled** in the same branches: model
   science, Derek's training work, Derek's explainer app, Concord's benchmark, and
   Concord's S3 deploy pipeline are all mixed together.

The goal is a fork whose diff against upstream is a small, legible set of genuinely
Concord-owned files, plus written instructions for the lab maintainer to bring the
upstream repo into the desired state.

## Background: the three-way fork chain

- **`upstream` = `Denolle-Lab/tiny-cnn-seismicML`** — the lab repo; source of truth for
  the model/training science.
- **`derek` = `yaoderek/tiny-cnn-seismicML`** — a collaborator's fork; source of the
  explainer app and the newer training notebooks. Derek's training improvements already
  reached upstream via PR #4; his explainer app and newest notebooks have not.
- **`origin` = `concord-consortium/tiny-cnn-seismicML`** — Concord's fork; staging ground
  for browser-readiness and S3 deployment of models for the CLUE Wave Runner tile. Per
  the CLUE design docs, CLUE consumes deployed model artifacts (metadata.json +
  weights.json) from S3 and has its own model-build code adapted from the repo.

Concord's relationship to the model is **consume now, maybe produce later**: the lab
trains models; Concord exports, deploys, and uses them — but wants to keep the door open
to training in the fork later.

## Current-state categorization (validated against the repos)

Everything that differs from upstream sorts into three buckets:

### Bucket A — Explainer app (Derek's → belongs upstream)
- `explainer-app/**` base (31 files; 24 match `derek/main` exactly)
- `scripts/export_waveforms_for_explainer.py` (matches `derek/main`)

### Bucket B — Training / data (Derek's → belongs upstream)
- `notebooks/02_labeling/download_AK_only_data.ipynb` — matches `derek/main`; **newer
  than upstream**
- `notebooks/03_training/train_cnn_multiclass.ipynb` — matches `derek/main`; **newer
  than upstream**
- `train.py`, `predict.py`, `scripts/export_to_browser.py` — minor diffs; effectively
  upstream's versions after `main` was merged into the clean branch (not Concord-owned).

### Bucket C — Concord-core (the genuinely Concord-owned layer)
- Deploy / consume:
  - `scripts/deploy-model.sh`
  - `scripts/export_compact_weights_for_tfjs.py`
  - `scripts/generate_dummy_weights.py`
  - `docs/deploy-model-for-clue.md`
  - `models/compact-v1/{metadata,weights}.json`
- Dev ergonomics:
  - `CLAUDE.md`
  - `.github/copilot-instructions.md`
  - `.github/workflows/ci.yml`
- Benchmark (Concord's, ~866 lines, **not** in Derek's repo):
  - `explainer-app/benchmark.html`, `explainer-app/src/benchmark.js`,
    `explainer-app/benchmark.md`, `explainer-app/apps-script/Code.gs`
  - Plus Concord's tweaks to 7 of Derek's explainer files (e.g. `seismicModel.js`).

## Target end-states

### Upstream / lab repo (lab maintainer brings it here)
- Model + training code *(already present)*
- Derek's newer notebooks + `export_waveforms_for_explainer.py` (from `derek`)
- The explainer app, Derek's base version (from `derek`)
- **Not** the benchmark — it embeds Concord's Google Sheet endpoint
  (`apps-script/Code.gs`, `benchmark.js`), which should not live in the lab repo.

### Concord fork
`main` = upstream + a thin Concord-only layer:

```
FORK.md                                   # NEW: fork relationship + sync/contribution rules
CLAUDE.md
.github/copilot-instructions.md
.github/workflows/ci.yml
docs/deploy-model-for-clue.md
scripts/deploy-model.sh                   # validation decoupled from explainer-app
scripts/export_compact_weights_for_tfjs.py
scripts/generate_dummy_weights.py
models/compact-v1/{metadata,weights}.json
benchmark/                                # NEW top-level home, self-contained
   benchmark.html
   benchmark.md
   src/benchmark.js
   src/<model-loader copy>                # own copy; no dependency on explainer app
   apps-script/Code.gs
   package.json                           # own deps (e.g. tfjs)
```

- **No explainer app maintained by Concord.** Dropped from Concord's branches; it lives
  in Derek's fork and (once adopted) upstream. Recoverable from git history.
- After cleanup, `git diff upstream/main main` on the fork shows ~10 Concord-owned paths
  instead of ~50.

### Derek's fork
Remains the source the lab pulls the explainer app + notebooks from. No change required
of Concord here beyond providing instructions to the lab maintainer.

## Coupling fixes required during cleanup

1. **`deploy-model.sh` validation.** It currently validates `metadata.json` by borrowing
   `ajv` from `explainer-app/node_modules`
   (`NODE_PATH=explainer-app/node_modules node -e "...ajv..."`). Replace with Python
   `jsonschema` (repo is Python-first) so the deploy script has no tie to the explainer
   app. The script already extracts the `$schema` URL and downloads the schema; only the
   validation call changes.

2. **`benchmark/` model loader.** `benchmark.js` imports the explainer's
   `seismicModel.js`. Since the explainer app leaves Concord's maintained tree, give the
   benchmark its own copy of the model-loading code so it is self-contained.

## Part A — Concord fork cleanup (mechanics)

The current PR branches entangle all buckets, so do not merge them. Instead:

1. Create a fresh `concord-core` branch off `main` (which already equals `upstream/main`).
2. Add only the Bucket C files to it:
   - Restore the deploy/dev files from `deploy-model-for-clue`.
   - Move the benchmark to top-level `benchmark/` and decouple it (coupling fix #2).
   - Apply coupling fix #1 to `deploy-model.sh`.
   - Add `FORK.md`.
3. Verify `examples/test_workflow.py` still passes and the deploy script validates a
   sample metadata file without the explainer app present.
4. Merge `concord-core` into `main`; push `origin main`.
5. Close PRs #2 and #3 as superseded, with a note pointing to the new `main`.
   Delete `browser-benchmark-clean` and `deploy-model-for-clue` (local + remote) once
   their wanted content is on `main`.
6. Resolve the parked `ai-instructions` branch (its `CLAUDE.md` /
   `copilot-instructions.md` are part of Bucket C and will be on `main`): delete it, or
   keep only if there is unique content.

## Part B — Instructions for the lab maintainer (to go in the spec / a handoff note)

Target: bring `Denolle-Lab/tiny-cnn-seismicML` up to date with Derek's explainer app and
newest training notebooks.

1. Add Derek's fork as a remote:
   `git remote add derek https://github.com/yaoderek/tiny-cnn-seismicML.git && git fetch derek`
2. Bring in the explainer app (Derek's base version) from `derek/main` — the
   `explainer-app/` directory and `scripts/export_waveforms_for_explainer.py`.
   - **Exclude** `explainer-app/benchmark.*`, `explainer-app/src/benchmark.js`, and
     `explainer-app/apps-script/Code.gs` if present — those are Concord's benchmark and
     point at a Concord Google Sheet.
3. Update the training notebooks to Derek's newer versions:
   `notebooks/02_labeling/download_AK_only_data.ipynb` and
   `notebooks/03_training/train_cnn_multiclass.ipynb`.
4. Open as one or more PRs for review; Concord can assist with the explainer-app PR if
   helpful.

After the lab adopts these, Concord syncs by merging `upstream/main` into the fork's
`main`; the explainer app then comes "for free" from upstream and never appears as a
Concord-specific diff.

## Risks / open items

- **Derek's notebooks newer than upstream:** confirm with Derek/lab that his versions are
  the intended ones before the lab pulls them.
- **Concord's explainer tweaks (7 files, e.g. `seismicModel.js`):** these were
  deploy-path / benchmark-integration changes. They are intentionally *not* carried
  upstream and *not* kept in the Concord fork (except the model-loader copy the benchmark
  needs). Confirm none are needed elsewhere before dropping.
- **Benchmark ↔ CLUE model code:** if the benchmark's model loader and CLUE's
  `buildCompactModel` should stay in sync, note where the canonical loader lives.
- **S3 preview deploys:** the old per-branch preview
  (`models-resources.concord.org/tiny-cnn-seismicML/branch/...`) may leave orphaned S3
  content after branch deletion — harmless, but note for cleanup.
```
