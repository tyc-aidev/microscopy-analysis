---
name: feature-development-pr-review
description: Runs the full feature development flow for this repo's two workstreams — Dataset Explorer (Streamlit) and MicroNet paper reproduction (PyTorch) — via worktree branches from main, uv env setup, tests, incremental commits, push, PR via gh cli, critical review, and follow-up issues. Use when the user wants a new feature or bugfix developed with a PR and review cycle.
---

# Feature Development with PR and Review

Follow this flow when the user asks for feature work that includes opening a PR and a review cycle (e.g. "develop a feature", "create a PR and get it reviewed", "do the full flow with review").

This repo has **two parallel workstreams**. Identify which one the PR belongs to before implementing:

| Workstream | Plan | Stack | GitHub milestone |
|---|---|---|---|
| **Dataset Explorer** (pre-Sprint-0) | [PLAN_DATASET_EXPLORER.md](../../../PLAN_DATASET_EXPLORER.md) | Streamlit, Pillow, pandas — **no PyTorch** | [#8](https://github.com/tyc-aidev/microscopy-analysis/issues/8) |
| **MicroNet paper reproduction** (Sprints 0–6) | [PLAN.md](../../../PLAN.md) | PyTorch, `segmentation_models_pytorch`, NASA `pretrained-microscopy-models` | [Milestone 1](https://github.com/tyc-aidev/microscopy-analysis/milestone/1) ([#1–#7](https://github.com/tyc-aidev/microscopy-analysis/issues)) |

Link PRs to the relevant issue. Do not mix explorer UI changes with training pipeline changes in one PR unless tightly coupled.

---

## MicroNet reproduction context (PyTorch)

When working on Sprints 0–6, follow [PLAN.md](../../../PLAN.md):

- **Goal:** Reproduce Stuckner et al. 2022 benchmark claims using [nasa/pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models) — not retrain MicroNet on the unreleased ~100k classification corpus.
- **Critical pitfall:** NASA `util.py` defaults to MicroNet **v1.1**. Paper reproduction must pin **`version=1.0`** everywhere (`pmm.util.get_pretrained_microscopynet_url(..., version=1.0)`).
- **Data:** 7 semantic benchmarks (`Super1–4`, `EBC1–3`) via sparse checkout of `benchmark_segmentation_data/`; optional Sprint 6 instance-seg COCO demo. Layout: `train`, `train_annot`, `val`, `val_annot`, `test`, `test_annot`.
- **DatasetAdapter rules:**
  - **Super:** `_mask.tif` pairing (NASA `io.py`); RGB masks — matrix `[0,0,0]`, secondary `[255,0,0]`, tertiary `[0,0,255]`
  - **EBC:** same filename in image/annot folders; grayscale `{0: bg, 1: oxide, 2: crack}`; binary task uses `{oxide: [1]}`
- **Models:** `create_segmentation_model()` with architectures UNet, UNet++, DeepLabV3+; pretraining regimes `random`, `imagenet`, `micronet`, `image-micronet`.
- **Training defaults** (match NASA notebooks before deviating): Adam `lr=2e-4` → phase 2 at `1e-5`; early stop `patience=30` on val IoU; `DiceBCELoss(weight=0.7)`; metric IoU@0.5; batch size 6; ImageNet normalization even for MicroNet encoders; patch inference 512/stride 256.
- **Project layout:** `src/amat/{data,models,train,eval,orchestration}/`, `configs/{datasets,experiments,cloud}/`, `scripts/{train,evaluate,run_matrix}.py`, `results/` (gitignored), `paper/target_metrics.csv`.
- **Dependency strategy:** `pip install git+https://github.com/nasa/pretrained-microscopy-models` + thin local wrapper; pin torch/smp to NASA [requirements_frozen.txt](https://github.com/nasa/pretrained-microscopy-models/blob/main/requirements_frozen.txt) in `requirements.txt` when added.

**Sprint exit criteria to verify in PR testing:**

| Sprint | Key check |
|---|---|
| 0 | Smoke train reproduces notebook IoU within ~2–5%; v1.0 weights load from NASA S3 |
| 1 | `python scripts/train.py --config configs/experiments/super1_baseline.yaml` → checkpoint + metrics JSON |
| 2 | Test IoU on all 7 datasets; MicroNet ≥ ImageNet on majority |
| 3 | Super3 low-data curve; relative IoU error reduction |
| 4 | Architecture + encoder sweep with seeds (mean ± std) |
| 5 | Patch inference, qualitative figs, reproduction report vs `paper/target_metrics.csv` |
| 6 | MMDetection instance-seg demo (optional) |

---

## 1. Worktree branch and implement

**Default:** create an isolated git worktree — do not check out the feature branch in the main repo working tree unless the user explicitly asks for in-place development.

```bash
git fetch origin main
BRANCH=feature/short-descriptive-name
WORKTREE=../amat-${BRANCH//\//-}   # e.g. ../amat-feature-dataset-explorer
git worktree add -b "$BRANCH" "$WORKTREE" origin/main
cd "$WORKTREE"
```

- All implementation, commits, and pushes happen inside the worktree directory.
- When done (after merge), remove the worktree: `git worktree remove "$WORKTREE"` (from any repo checkout).
- **In-place fallback** (only when requested or worktrees are unavailable):
  ```bash
  git fetch origin main
  git checkout -b feature/short-descriptive-name origin/main
  ```

### Environment

Use [uv](https://docs.astral.sh/uv/) with **Python 3.12** (see [README.md](../../../README.md)):

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

Install the right dependency set for the workstream:

| Workstream | Requirements file | Setup notes |
|---|---|---|
| Dataset Explorer | `requirements-explorer.txt` | `./scripts/setup_env.sh` installs this automatically |
| MicroNet reproduction | `requirements.txt` (when added) | CUDA PyTorch pinned to NASA frozen env; may need a separate `./scripts/setup_env.sh` variant or manual `uv pip install -r requirements.txt` after venv creation |

For reproduction work, also ensure benchmark data is present (`scripts/download_data.sh` or sparse checkout per PLAN.md).

### Implement

- Implement in logical chunks. Create a todo list for multi-step work.
- **Explorer:** `explorer/` (Streamlit app, pages, `lib/`), `explorer/datasets/catalog.json`
- **Reproduction:** `src/amat/`, `configs/`, `scripts/train.py`, etc. per PLAN.md structure
- Write or update tests for pure Python logic (`explorer/lib/`, `src/amat/data/`, etc.)

### Run tests and manual checks

**Unit / integration tests:**

```bash
pytest path/to/test_file.py -q
pytest tests/ -q
```

**Dataset Explorer (manual):**

```bash
streamlit run explorer/app.py
```

Verify pages load, filters work, mask overlays render correctly, and `DATA_ROOT` / missing-data prompts behave as expected.

**MicroNet reproduction (when touching ML code):**

```bash
python -c "import torch; print(torch.cuda.is_available())"
python scripts/train.py --config configs/experiments/super1_baseline.yaml
python scripts/evaluate.py --config configs/experiments/super1_baseline.yaml   # when available
```

Run only what the change touches — explorer-only PRs should not require CUDA or PyTorch.

## 2. Incremental commits

- One commit per logical change (e.g. lib module, Streamlit page, DatasetAdapter, trainer, config, tests).
- Push branch from the worktree:
  ```bash
  git push -u origin feature/branch-name
  ```

## 3. Create PR with gh cli

- Create PR with title and body:
  ```bash
  gh pr create --base main --head feature/branch-name --title "feat(scope): short title" --body "## Summary
  ...
  ## Testing
  ...
  ## Commits
  - ..."
  ```
- Body: summary of changes, workstream (explorer vs reproduction), linked issue, what was tested, and list of commits.

## 4. Review as critical reviewer

Switch to production gatekeeper mindset. Review for:

- **Explorer correctness:** mask pairing (Super `_mask.tif` vs EBC same-filename), class colors/values, COCO parsing, split counts including `different_test` on Super2/3
- **Streamlit:** `@st.cache_data` keys include inputs that affect output; `st.session_state` reset when filters/sort change; no blocking I/O on every rerun
- **Reproduction correctness:** Super vs EBC `DatasetAdapter` paths; pretraining regime wiring; two-phase schedule + early stopping; IoU computed on held-out test set (not val leakage)
- **PyTorch:** MicroNet **`version=1.0`** everywhere; device handling (CPU fallback in unit tests); no silent shape mismatches; checkpoint resume; results in `results/` not committed to git
- **Hyperparameters:** changes from NASA defaults must be justified in config comments or PR description
- **General:** consistency with NASA `io.py` conventions, maintainability, type hints where the module already uses them

Submit review via GitHub API:

```bash
gh api repos/OWNER/REPO/pulls/PR_NUMBER/reviews -f commit_id=COMMIT_SHA -f event=COMMENT -f body="**Review**\n\n..."
```

Get `COMMIT_SHA` with: `gh pr view PR_NUMBER --json commits -q '.commits[-1].oid'`

If inline review comments fail ("Line could not be resolved"), put file/line and suggestion text in the review body instead.

## 5. Address review as developer

- Implement suggested changes in the **same worktree**.
- Re-run relevant checks (`pytest`, `streamlit run`, smoke train/eval), then:
  ```bash
  git add ...
  git commit -m "fix(scope): address PR review - ..."
  git push
  gh pr comment PR_NUMBER --body "Addressed ... (commit SHA). Ready for re-review."
  ```

## 6. File follow-up issues (before merge)

When review defers work (v2, cloud orchestration, UI polish, full encoder sweep), open GitHub issues so nothing is lost.

- One issue per distinct follow-up (not one mega-issue).
- Title: `area: short outcome` (e.g. `train: cloud job queue for Sprint 2 matrix`, `explorer: aggregate stats charts on Home page`).
- Body: context from the PR/review, goal, acceptance criteria, links to PR, PLAN.md sprint, and relevant paths.
- Check for duplicates: `gh issue list --search "keywords"`.
- Create:
  ```bash
  gh issue create --repo OWNER/REPO --title "area: ..." --body "$(cat <<'EOF'
  ## Context
  ...
  ## Goal
  ...
  ## Acceptance criteria
  - [ ] ...
  EOF
  )"
  ```
- Comment on the PR with issue URLs and note them in the PR description if helpful.

## Notes

- Prefer `gh pr create`, `gh pr view`, `gh pr comment`, and `gh api .../pulls/.../reviews` so the whole flow stays in the repo and terminal.
- Branch naming: `feature/...`, `bugfix/...`, `refactor/...`.
- Worktree naming: sibling directory `../amat-<branch-with-slashes-as-dashes>` keeps the main checkout on `main`.
- Keep explorer deps (`requirements-explorer.txt`) separate from the PyTorch stack unless the PR genuinely needs both.
- Reproduction PRs should reference the sprint issue and cite which exit criteria from PLAN.md were verified.
