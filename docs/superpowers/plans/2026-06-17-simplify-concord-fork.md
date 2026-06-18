# Simplify the Concord Fork — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the Concord fork's diff against upstream to a small, legible set of Concord-owned files by isolating Concord-core, making the benchmark self-contained, dropping the explainer app, and documenting the fork + a lab-maintainer handoff.

**Architecture:** Build a fresh `concord-core` branch off `main` (which already equals `upstream/main`), populate it with only the Concord-owned files (deploy/consume tooling, dev-ergonomics files, and a decoupled top-level `benchmark/`), then merge it to `main`. Close the entangled PRs #2/#3 and delete their branches. The explainer app and Derek's notebooks are handed to the lab via written instructions, not carried in the fork.

**Tech Stack:** git, bash, Python 3 (`jsonschema`), Node/Vite + TensorFlow.js (benchmark only), AWS CLI (deploy, not exercised here).

## Global Constraints

- Code style: Google-style docstrings, **no type hints** (Python); follow existing file conventions.
- The repo has **no unit-test suite**; `examples/test_workflow.py` is the integration sanity check.
- `main` tracks `origin/main`; `origin` = `concord-consortium/tiny-cnn-seismicML`, `upstream` = `Denolle-Lab/tiny-cnn-seismicML`, `derek` = `yaoderek/tiny-cnn-seismicML`.
- Do not push to `origin` or delete remote branches until the final task (Task 8), and confirm before each outward-facing action.
- Source of all Concord-core content: the `deploy-model-for-clue` branch (it contains every Bucket C file).
- The design spec lives (untracked) at `docs/superpowers/specs/2026-06-17-simplify-concord-fork-design.md`; this plan lives (untracked) at `docs/superpowers/plans/2026-06-17-simplify-concord-fork.md`. Both get committed onto `concord-core` in Task 7 — never onto a branch slated for deletion.

---

### Task 1: Create `concord-core` and bring over deploy + dev files

**Files:**
- Create branch: `concord-core` (off `main`)
- Restore from `deploy-model-for-clue`:
  - `scripts/deploy-model.sh`
  - `scripts/export_compact_weights_for_tfjs.py`
  - `scripts/generate_dummy_weights.py`
  - `docs/deploy-model-for-clue.md`
  - `models/compact-v1/metadata.json`
  - `models/compact-v1/weights.json`
  - `CLAUDE.md`
  - `.github/copilot-instructions.md`
  - `.github/workflows/ci.yml`

**Interfaces:**
- Produces: a `concord-core` branch whose only diff vs `main` is the deploy/dev file set above (no explainer app, no benchmark yet).

- [ ] **Step 1: Create the branch off main**

```bash
git checkout main
git checkout -b concord-core
```

- [ ] **Step 2: Pull the deploy + dev files from deploy-model-for-clue**

```bash
git checkout deploy-model-for-clue -- \
  scripts/deploy-model.sh \
  scripts/export_compact_weights_for_tfjs.py \
  scripts/generate_dummy_weights.py \
  docs/deploy-model-for-clue.md \
  models/compact-v1/metadata.json \
  models/compact-v1/weights.json \
  CLAUDE.md \
  .github/copilot-instructions.md \
  .github/workflows/ci.yml
```

- [ ] **Step 3: Verify the staged set is exactly these paths**

Run: `git status --short`
Expected: only the 9 paths above appear as added/modified (`A`/`M`). No `explainer-app/` entries.

- [ ] **Step 4: Sanity-check Python still imports/runs**

Run: `python examples/test_workflow.py`
Expected: completes without import errors (exit 0).

- [ ] **Step 5: Commit**

```bash
git add scripts/deploy-model.sh scripts/export_compact_weights_for_tfjs.py scripts/generate_dummy_weights.py docs/deploy-model-for-clue.md models/compact-v1/ CLAUDE.md .github/copilot-instructions.md .github/workflows/ci.yml
git commit -m "Add Concord-core deploy and dev files"
```

---

### Task 2: Decouple deploy validation from the explainer app

Replace the `ajv` / `explainer-app/node_modules` validation in `deploy-model.sh` with a small Python validator using `jsonschema`, so the deploy script has no dependency on the explainer app.

**Files:**
- Create: `scripts/validate-metadata.py`
- Modify: `scripts/deploy-model.sh:49-58` (the `NODE_PATH=... node -e ...` block)
- Modify: `requirements.txt` (add `jsonschema`)

**Interfaces:**
- Produces: `scripts/validate-metadata.py <schema_file> <metadata_file>` — exits 0 and prints `Schema validation passed.` on success; prints errors and exits 1 on failure.

- [ ] **Step 1: Write the validator with a self-test docstring example**

Create `scripts/validate-metadata.py`:

```python
"""Validate a model metadata JSON file against a JSON Schema.

Usage:
    python scripts/validate-metadata.py <schema_file> <metadata_file>

Exits 0 and prints "Schema validation passed." when the metadata satisfies
the schema; prints the validation errors and exits 1 otherwise.
"""

import json
import sys

import jsonschema


def main(schema_path, metadata_path):
    """Validate metadata against schema.

    Args:
        schema_path: Path to the JSON Schema file.
        metadata_path: Path to the metadata JSON file to validate.

    Returns:
        0 if valid, 1 if invalid.
    """
    with open(schema_path) as f:
        schema = json.load(f)
    with open(metadata_path) as f:
        data = json.load(f)

    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        print("Schema validation failed:")
        for err in errors:
            location = "/".join(str(p) for p in err.path) or "(root)"
            print(f"  {location}: {err.message}")
        return 1

    print("Schema validation passed.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/validate-metadata.py <schema_file> <metadata_file>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 2: Add the dependency**

Add this line to `requirements.txt`:

```
jsonschema>=4.0.0
```

Run: `pip install 'jsonschema>=4.0.0'`
Expected: installs (or "already satisfied").

- [ ] **Step 3: Verify the validator passes on the real metadata against a known-good local schema**

The metadata's `$schema` points at the live CLUE schema URL. Test offline with a minimal schema that the current metadata satisfies:

```bash
cat > /tmp/seismic-schema.json <<'JSON'
{
  "type": "object",
  "required": ["id", "architecture", "class_names", "sampling_rate", "window_duration", "weightsUrl"],
  "properties": {
    "id": {"type": "string"},
    "architecture": {"type": "string"},
    "class_names": {"type": "array", "items": {"type": "string"}},
    "sampling_rate": {"type": "number"},
    "window_duration": {"type": "number"},
    "weightsUrl": {"type": "string"}
  }
}
JSON
python scripts/validate-metadata.py /tmp/seismic-schema.json models/compact-v1/metadata.json
```

Expected: prints `Schema validation passed.` (exit 0).

- [ ] **Step 4: Verify it fails on bad input**

```bash
echo '{"id": 123}' > /tmp/bad-metadata.json
python scripts/validate-metadata.py /tmp/seismic-schema.json /tmp/bad-metadata.json; echo "exit=$?"
```

Expected: prints `Schema validation failed:` with errors and `exit=1`.

- [ ] **Step 5: Replace the node/ajv block in deploy-model.sh**

In `scripts/deploy-model.sh`, replace lines 49-58 (the `NODE_PATH=explainer-app/node_modules node -e "..."` block, ending at the line with `" "$SCHEMA_FILE" "$METADATA_FILE"`) with:

```bash
python3 scripts/validate-metadata.py "$SCHEMA_FILE" "$METADATA_FILE"
```

- [ ] **Step 6: Confirm no explainer-app reference remains**

Run: `grep -n "explainer-app" scripts/deploy-model.sh; echo "matches=$?"`
Expected: no output, `matches=1` (grep found nothing).

- [ ] **Step 7: Commit**

```bash
git add scripts/validate-metadata.py scripts/deploy-model.sh requirements.txt
git commit -m "Decouple deploy validation from explainer-app (use Python jsonschema)"
```

---

### Task 3: Extract the benchmark into a self-contained top-level `benchmark/` and repurpose CI to deploy it

The benchmark currently lives under `explainer-app/` and imports the explainer's model loader. Give it its own copy of `seismicModel.js` (which imports only `@tensorflow/tfjs`), its own compact weights, its own Vite/package config, and repurpose the S3-deploy CI workflow to build and deploy the benchmark (instead of the explainer app, which is leaving the fork).

Important facts (verified against the repo):
- `compact_weights.json` IS git-tracked; copy it. `standard_weights.json` is **gitignored** (a generated dummy) — do NOT copy it; generate it via `scripts/generate_dummy_weights.py` and gitignore it under `benchmark/`.
- `scripts/generate_dummy_weights.py` hardcodes its default output dir to `explainer-app/public/models` (variable `MODELS_DIR`, ~line 20) but accepts `--output_dir`. Point the default at `benchmark/public/models`.
- `.github/workflows/ci.yml` runs on every push: it generates standard dummy weights, `npm ci` in `explainer-app`, then deploys via `concord-consortium/s3-deploy-action` with `workingDirectory: explainer-app`. Repurpose both `working-directory`/`workingDirectory` to `benchmark`.

**Files:**
- Create (from `deploy-model-for-clue`'s `explainer-app/`):
  - `benchmark/index.html`            (from `explainer-app/benchmark.html`)
  - `benchmark/src/benchmark.js`      (from `explainer-app/src/benchmark.js`)
  - `benchmark/src/lib/seismicModel.js` (from `explainer-app/src/lib/seismicModel.js`)
  - `benchmark/public/models/compact_weights.json`  (tracked; from `explainer-app/public/models/compact_weights.json`)
  - `benchmark/apps-script/Code.gs`   (from `explainer-app/apps-script/Code.gs`)
  - `benchmark/benchmark.md`          (from `explainer-app/benchmark.md`)
- Create new: `benchmark/package.json`, `benchmark/vite.config.js`, `benchmark/.gitignore`
- Modify: `scripts/generate_dummy_weights.py` (default output dir → `benchmark/public/models`)
- Modify: `.github/workflows/ci.yml` (deploy `benchmark/` instead of `explainer-app/`)
- Generated, NOT committed: `benchmark/public/models/standard_weights.json` (gitignored)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: a `benchmark/` app that builds with `npm install && npm run build` independent of `explainer-app/`, deployed by `ci.yml` on push.

- [ ] **Step 1: Materialize the benchmark source files from deploy-model-for-clue into their new paths**

```bash
mkdir -p benchmark/src/lib benchmark/public/models benchmark/apps-script
git show deploy-model-for-clue:explainer-app/benchmark.html              > benchmark/index.html
git show deploy-model-for-clue:explainer-app/src/benchmark.js            > benchmark/src/benchmark.js
git show deploy-model-for-clue:explainer-app/src/lib/seismicModel.js     > benchmark/src/lib/seismicModel.js
git show deploy-model-for-clue:explainer-app/public/models/compact_weights.json  > benchmark/public/models/compact_weights.json
git show deploy-model-for-clue:explainer-app/apps-script/Code.gs         > benchmark/apps-script/Code.gs
git show deploy-model-for-clue:explainer-app/benchmark.md                > benchmark/benchmark.md
```

(Note: `standard_weights.json` is intentionally not copied — it is generated in Step 6.)

- [ ] **Step 2: Confirm the script tag and imports resolve within benchmark/**

Run: `grep -n 'src/benchmark.js' benchmark/index.html; grep -n "from './lib/seismicModel.js'" benchmark/src/benchmark.js; grep -nE "from ['\"]" benchmark/src/lib/seismicModel.js`
Expected: `index.html` references `/src/benchmark.js`; `benchmark.js` imports `./lib/seismicModel.js`; `seismicModel.js` imports only `@tensorflow/tfjs`. (All paths now exist under `benchmark/`.)

- [ ] **Step 3: Create benchmark/package.json**

```json
{
  "name": "seismic-benchmark",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@tensorflow/tfjs": "^4.22.0",
    "@tensorflow/tfjs-backend-webgpu": "^4.22.0"
  },
  "devDependencies": {
    "vite": "^7.3.1"
  }
}
```

- [ ] **Step 4: Create benchmark/vite.config.js**

```javascript
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  base: './',
})
```

- [ ] **Step 5: Create benchmark/.gitignore**

The standard weights are a generated dummy and must not be committed (mirrors the explainer-app gitignore rule).

```
node_modules
dist

# Generated dummy weights for benchmarking (regenerate with scripts/generate_dummy_weights.py)
public/models/standard_weights.json
```

- [ ] **Step 6: Point generate_dummy_weights.py at the benchmark and generate the standard weights**

In `scripts/generate_dummy_weights.py`, change the default output directory (the `MODELS_DIR` assignment, ~line 20) from the `explainer-app` path to the benchmark path:

```python
MODELS_DIR = REPO_ROOT / "benchmark" / "public" / "models"
```

Then generate the standard dummy weights locally so the build can find them:

```bash
pip install numpy >/dev/null 2>&1 || true
python scripts/generate_dummy_weights.py --model standard
```

Expected: prints `Generated standard dummy weights ... -> .../benchmark/public/models/standard_weights.json` and the file exists. Confirm it is gitignored:
Run: `git check-ignore benchmark/public/models/standard_weights.json`
Expected: prints the path (it is ignored).

- [ ] **Step 7: Verify the benchmark builds with no explainer-app dependency**

```bash
cd benchmark && npm install && npm run build && cd ..
```

Expected: `npm run build` completes; `benchmark/dist/index.html`, `benchmark/dist/models/compact_weights.json`, and `benchmark/dist/models/standard_weights.json` all exist.

Run: `grep -rn "explainer-app" benchmark/ --include='*.js' --include='*.html' --include='*.json' --include='*.md' | grep -v node_modules | grep -v dist; echo "matches=$?"`
Expected: no output, `matches=1`.

- [ ] **Step 8: Repurpose ci.yml to deploy the benchmark**

In `.github/workflows/ci.yml`, change the two `explainer-app` references to `benchmark` so CI builds and deploys the benchmark app. After editing, the relevant lines read:

```yaml
      - name: Install Dependencies
        working-directory: benchmark
        run: npm ci
```

and

```yaml
      - uses: concord-consortium/s3-deploy-action@v1
        with:
          bucket: models-resources
          prefix: tiny-cnn-seismicML
          workingDirectory: benchmark
          githubToken: ${{ secrets.GITHUB_TOKEN }}
          deployRunUrl: https://models-resources.concord.org/tiny-cnn-seismicML/__deployPath__/index.html
          topBranches: |
            ["main"]
```

Leave the "Generate standard dummy weights" step as `python scripts/generate_dummy_weights.py --model standard` — it now writes to `benchmark/public/models` via the new default from Step 6.

Verify no explainer-app reference remains in the workflow:
Run: `grep -n "explainer-app" .github/workflows/ci.yml; echo "matches=$?"`
Expected: no output, `matches=1`.

- [ ] **Step 9: Commit**

```bash
git add benchmark/index.html benchmark/src benchmark/public benchmark/apps-script benchmark/benchmark.md benchmark/package.json benchmark/vite.config.js benchmark/.gitignore scripts/generate_dummy_weights.py .github/workflows/ci.yml
git commit -m "Move benchmark to self-contained benchmark/ and deploy it via CI"
```

---

### Task 4: Confirm the tree is clean of the explainer app

`concord-core` was branched from `main` (= upstream), which never had `explainer-app/`, so there is nothing to delete — this task verifies that and that everything still works without it.

**Files:** none modified.

- [ ] **Step 1: Verify explainer-app is absent from the branch**

Run: `git ls-files | grep -c '^explainer-app/' ; ls explainer-app 2>/dev/null && echo "PRESENT" || echo "absent"`
Expected: `0` tracked files; working tree `absent` (if a stray untracked `explainer-app/` remains from the prior branch, remove it: `rm -rf explainer-app`).

- [ ] **Step 2: Verify deploy validation works without explainer-app present**

```bash
python scripts/validate-metadata.py /tmp/seismic-schema.json models/compact-v1/metadata.json
```

Expected: `Schema validation passed.`

- [ ] **Step 3: Verify the benchmark still builds without explainer-app present**

```bash
cd benchmark && npm run build && cd ..
```

Expected: build succeeds.

- [ ] **Step 4: No commit needed** (verification-only task). If Step 1 required removing a stray dir, there is nothing tracked to commit.

---

### Task 5: Write FORK.md

Document the fork relationship and conventions so a new developer understands what Concord owns and how syncing/contributing works.

**Files:**
- Create: `FORK.md`

- [ ] **Step 1: Write FORK.md**

```markdown
# About this fork

`concord-consortium/tiny-cnn-seismicML` is a fork of
[`Denolle-Lab/tiny-cnn-seismicML`](https://github.com/Denolle-Lab/tiny-cnn-seismicML)
(the "lab" repo), the source of truth for the seismic CNN model and training code.

Concord's role is to **consume** trained models — export them to TensorFlow.js
weights, deploy them to S3, and use them in the CLUE Wave Runner tile — with the
option to train in this fork later.

## Remotes

| Remote     | Repo                                   | Role                                   |
|------------|----------------------------------------|----------------------------------------|
| `origin`   | `concord-consortium/tiny-cnn-seismicML`| This fork                              |
| `upstream` | `Denolle-Lab/tiny-cnn-seismicML`       | The lab repo (model/training science) |
| `derek`    | `yaoderek/tiny-cnn-seismicML`          | Collaborator; source of the explainer app |

## What Concord owns (everything else comes from upstream)

- `scripts/deploy-model.sh`, `scripts/validate-metadata.py` — deploy model files to S3
- `scripts/export_compact_weights_for_tfjs.py`, `scripts/generate_dummy_weights.py` — export helpers
- `docs/deploy-model-for-clue.md` — deploy guide
- `models/<model-id>/` — deployed model artifacts (metadata + weights)
- `benchmark/` — self-contained browser benchmark (uploads to a Concord Google Sheet); built and deployed to S3 on push by `.github/workflows/ci.yml`
- `CLAUDE.md`, `.github/copilot-instructions.md` — dev ergonomics
- `.github/workflows/ci.yml` — builds and deploys `benchmark/` to S3
- `FORK.md` — this file

Run `git diff upstream/main main` to see Concord's additions at a glance.

## Syncing from the lab

```bash
git fetch upstream
git checkout main
git merge upstream/main      # fast-forward or a small merge; resolve as needed
git push origin main
```

## Contributing upstream

Open PRs against `upstream` from a branch based on `upstream/main` that contains
**only** generally-useful changes (model/training improvements). Do **not** include
Concord-only paths listed above — especially `benchmark/` (it points at a Concord
spreadsheet) and the S3 deploy tooling (Concord infrastructure).

## The explainer app

The browser explainer app is maintained in Derek's fork and is being adopted by the
lab repo. It is intentionally **not** maintained in this fork. If you need it, get it
from `upstream` (after the lab adopts it) or from `derek/main`.
```

- [ ] **Step 2: Commit**

```bash
git add FORK.md
git commit -m "Add FORK.md documenting fork relationship and conventions"
```

---

### Task 6: Write the lab-maintainer handoff instructions

**Files:**
- Create: `docs/upstream-handoff.md`

- [ ] **Step 1: Write docs/upstream-handoff.md**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/upstream-handoff.md
git commit -m "Add lab-maintainer upstream handoff instructions"
```

---

### Task 7: Commit the design spec and this plan onto concord-core

The spec and plan are currently untracked (created on a to-be-deleted branch). Track them now that we have a durable branch.

**Files:**
- Add: `docs/superpowers/specs/2026-06-17-simplify-concord-fork-design.md`
- Add: `docs/superpowers/plans/2026-06-17-simplify-concord-fork.md`

- [ ] **Step 1: Verify both files exist in the working tree**

Run: `ls docs/superpowers/specs/2026-06-17-simplify-concord-fork-design.md docs/superpowers/plans/2026-06-17-simplify-concord-fork.md`
Expected: both paths listed.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-17-simplify-concord-fork-design.md docs/superpowers/plans/2026-06-17-simplify-concord-fork.md
git commit -m "Add fork-simplification design spec and implementation plan"
```

---

### Task 8: Merge to main, push, close PRs, delete superseded branches

Final consolidation. This is the only outward-facing task — confirm before each push/close/delete.

**Files:** none (git operations).

- [ ] **Step 1: Review the full Concord-core diff vs upstream**

Run: `git checkout concord-core && git diff --stat upstream/main concord-core`
Expected: only Concord-owned paths (Task 1 files, `scripts/validate-metadata.py`, `requirements.txt`, `benchmark/**`, `FORK.md`, `docs/upstream-handoff.md`, `docs/superpowers/**`). No `explainer-app/`, no notebook changes.

- [ ] **Step 2: Merge concord-core into main**

```bash
git checkout main
git merge --no-ff concord-core -m "Simplify fork: Concord-core only (deploy, benchmark, docs)"
```

- [ ] **Step 3: Verify main's diff vs upstream is the Concord-core set**

Run: `git diff --name-only upstream/main main`
Expected: the Concord-owned paths only.

- [ ] **Step 4: Push main (confirm first)**

```bash
git push origin main
```

- [ ] **Step 5: Close PRs #2 and #3 as superseded**

```bash
gh pr close 2 --repo concord-consortium/tiny-cnn-seismicML --comment "Superseded: Concord-core work has been consolidated onto main, with the explainer app handed to the lab repo (see docs/upstream-handoff.md). Closing."
gh pr close 3 --repo concord-consortium/tiny-cnn-seismicML --comment "Superseded: deploy tooling is now on main as part of the fork simplification. Closing."
```

- [ ] **Step 6: Delete superseded branches (local + remote)**

```bash
git branch -D browser-benchmark-clean deploy-model-for-clue ai-instructions
git push origin --delete browser-benchmark-clean deploy-model-for-clue
```

(`ai-instructions` was never pushed to `origin`; its `CLAUDE.md` / `copilot-instructions.md` are now on `main`, so the local delete is sufficient. Verify with `git branch -a` that no unexpected branches remain.)

- [ ] **Step 7: Final verification**

Run: `git branch && echo '---' && git diff --name-only upstream/main main`
Expected: branches are `main` and `concord-core` (delete `concord-core` too if you prefer, since it's merged: `git branch -d concord-core`). The diff lists only Concord-owned paths.

---

## Self-Review notes

- **Spec coverage:** Target fork layout → Tasks 1,3,5,6,7; coupling fix #1 (deploy validation) → Task 2; coupling fix #2 (benchmark loader) → Task 3; Part A cleanup mechanics → Tasks 1,4,8; Part B lab instructions → Task 6; "no explainer app in fork" → Task 4 (verified absent because branched from upstream). All spec sections mapped.
- **Risk — Derek's notebooks:** surfaced in Task 6 Step 3 ("confirm with Derek").
- **Risk — orphaned S3 previews:** out of scope here; noted in the spec's Risks section.
</content>
