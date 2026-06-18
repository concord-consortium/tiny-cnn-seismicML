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
