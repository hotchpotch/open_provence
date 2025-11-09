# AI Agent Operating Guide

## Repository Map
- Core pruning and ranking logic lives in `open_provence/`, with `encoder.py`, `trainer.py`, `models/open_provence_head.py`, and `modeling_open_provence_standalone.py` coordinating model internals; helpers sit under `open_provence/utils/`.
- CLI entry points live in `scripts/`: training/evaluation wrappers such as `eval_datasets.py` and `eval_mldr.py`, dataset converters under `context-relevance-datasets/`, and release helpers under `utils/`. Launch them via `uv run python ...` or the `open_provence_trainer` console script.
- Experiment definitions live in `configs/` (YAML/JSON). Generated artefacts belong in `output/<config>_<timestamp>/` and mirrored to `wandb/`; keep ad-hoc scratch files in `tmp/`.
- Documentation is under `docs/` (see @docs/train.md, @docs/create_context_relevance_dataset.md, @docs/eval_dataset.md, @docs/eval_mldr.md, `@docs/eval_reports/*`). Tests mirror the package within `tests/`.
- Python baseline is 3.11+. Respect Ruff defaults (4-space indent, ≤99 characters, sorted imports) and start new modules with `from __future__ import annotations`.

## Architecture Overview
- **Data + Config Layer**: YAML/JSON configs in `configs/` describe datasets, training hyperparameters, and evaluation suites. Dataset preparation scripts under `scripts/context-relevance-datasets/` convert raw triplets into Provence format (@docs/create_context_relevance_dataset.md).
- **Training Stack**: `python -m open_provence.runner` / `open_provence_trainer` parse configs into `open_provence/trainer.py` and `open_provence/encoder.py`, which orchestrate model heads from `open_provence/models/`. The workflow, checkpoints, and reference metrics are documented in @docs/train.md.
- **Model Packaging**: `open_provence/modeling_open_provence_standalone.py` is the Hugging Face facing entry point. It must stay dependency-light and portable because checkpoints exported to the Hub ship this file for remote inference; preserve its standalone imports and ensure inference paths remain functional.
- **Inference & Serving**: Local inference relies on `modeling_open_provence_standalone.py` (auto-loaded by `AutoModel.from_pretrained(..., trust_remote_code=True)`) or the evaluation CLIs (`scripts/eval_datasets.py`, `scripts/eval_mldr.py`) that wrap `model.process` for batch pruning. Downstream agents integrate via the same API shown in README examples.
- **Evaluation Layer**: Dataset retention metrics live in `scripts/eval_datasets.py` (@docs/eval_dataset.md), while long-document robustness uses `scripts/eval_mldr.py` (@docs/eval_mldr.md). Generated reports are stored under `output/.../eval_results/` or `@docs/eval_reports/*`.
- **Observability**: Training metadata is mirrored to `wandb/` and organised by run slug (`<config>-<timestamp>`). Logs and summaries (Markdown/JSON) accompany each run for regression tracking.
- **Release Flow**: Successful checkpoints move from `output/<config>_<timestamp>/final_model/` into `output/release_models/<name>/`, bundling the standalone modeling file and evaluation artefacts before publishing to Hugging Face.

## Environment Setup
- Install dependencies with `uv sync`. By default this now resolves the CUDA 12.8 wheel (`torch==2.7.1+cu128`) on Linux x86_64; use `uv sync --no-default-groups --group dev --group cpu` when you need the CPU/Metal build instead.
- If you need to pin a specific CUDA wheel manually after syncing, run:
  ```bash
  uv pip install --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match "torch==2.7.1+cu128"
  ```
  Optional: install FlashAttention via `uv sync --group flash-attn` or a vetted wheel in `tmp/`.
- Export credentials used by the tooling: `WANDB_API_KEY`, Hugging Face token (`huggingface-cli login`), and (for MLDR LLM judging) `OPENAI_API_KEY`.
- Use local virtualenv isolation (`uv`) and avoid polluting the repo with global cache files; large downloads belong under `output/`, `wandb/`, or `tmp/`.

## Day-to-Day Development
- Run the full CI stack with `uv run tox run-parallel` after implementing changes; it runs pytest, Ruff, formatting, and type checks in one go while keeping the environments concurrent. When you are instructed to “run tox” locally, default to `uv run tox run-parallel` to complete the suite faster.
- For focused iteration, the equivalent single commands are:
  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check --diff
  uv run pyright
  ```
- Launch training from configs with either `uv run python -m open_provence.runner <config.yaml>` or `uv run open_provence_trainer <config.yaml>`.
- Run pruning sweeps via `uv run python scripts/eval_datasets.py --config configs/eval_datasets/en_nano.yaml --model output/<run>/final_model --threshold 0.1 --output-json tmp/eval_en_nano.json --output-file tmp/eval_en_nano.md`; use `scripts/eval_mldr.py` for MLDR runs (flags in docs).
- Keep notebooks and diagnostics in `tmp/` and mirror new helpers under `tests/` to maintain parity with production modules.

## Regression Guardrails for Modeling Changes
- If you touch training loops, heads, or model plumbing (e.g., files under `open_provence/models/`, `open_provence/trainer.py`, `open_provence/modeling_open_provence_standalone.py`), reproduce the reference runs from `docs/train.md`:
  ```bash
  uv run open_provence_trainer configs/toy-open-provence-reranker-v1-gte-modernbert-base.yaml
  uv run open_provence_trainer configs/toy-open-provence-reranker-v1.yaml
  ```
  These complete in roughly 5 minutes (English) and about 6 minutes (Japanese, including nano
  evals) on an RTX 4090; record the output directories and compare nano metrics against the
  documented baselines (F2 ≈ 0.80 en / 0.79 ja at threshold 0.1).
- Archive artefacts under `output/<config>_<timestamp>/final_model/` and sync Weights & Biases runs (`toy-open-provence-reranker-*-20251029-*`) for traceability.
- When PRs alter model behaviour, summarize the training command, threshold metrics, and any evaluation deltas directly in the PR body, citing the artefacts. Block merges if the nano checks regress materially.

## Evaluation Pipelines
- Dataset-level pruning checks: follow `docs/eval_dataset.md` and run
  ```bash
  uv run python scripts/eval_datasets.py \
    --config configs/eval_datasets/en_nano.yaml \
    --model output/<run>/final_model \
    --threshold 0.1 \
    --batch-size 256 \
    --output-json tmp/eval_en_nano.json \
    --output-file tmp/eval_en_nano.md
  ```
  Swap in `ja_nano.yaml` or the full configs once metrics look good. Copy JSON/MD summaries into the release tree when publishing models.
- MLDR long-document evaluation: see `docs/eval_mldr.md`. Typical invocation:
  ```bash
  uv run python scripts/eval_mldr.py \
    --model output/<run>/final_model \
    --lang en \
    --threshold 0.1 \
    --limit 200 \
    --ignore-file scripts/eval_mldr/ignored_questions.yaml \
    --output-dir output/eval_mldr_runs/<slug>
  ```
  Preserve `process/` and `eval_llm/summary.{json,md}` outputs, and document any ignore-list edits.

## Dataset Creation Workflow
- Convert triplet datasets using `scripts/context-relevance-datasets/generate_ds_from_sentense_transformer.py` (see `docs/create_context_relevance_dataset.md` for flags and prerequisites).
- Augment context spans with relevance labels (`add_context_spans_relevance.py`) and teacher scores (`add_reranker_teacher_scores.py`); install language-specific splitters (`nltk`, `fast-bunkai`) beforehand.
- Verify conversions with small previews (`datasets.load_from_disk`) and keep intermediate artefacts under `output/context-relevance-datasets/`.
- When sharing new datasets, document schema quirks or filters in `docs/` and reference the Hugging Face IDs used.

## Commit, Review, and PR Expectations
- Commit messages follow `type: concise summary` (e.g., `feat: add bilingual pruning thresholds`) and reference relevant configs or issues.
- Before opening a PR:
  - Run `uv run tox` (or `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check --diff`, `uv run pyright`) and include the outcomes.
  - Document manual diff reviews (`git diff`, `git diff origin/main...`) and summarize affected scripts/configs plus any artefacts written to `output/` or `wandb/`.
  - Update docs when behaviour changes (e.g., amend `docs/train.md` if new training steps are required).
  - Attach evaluation results (nano dataset metrics, MLDR summaries) whenever model quality could shift.
- Keep new assets in `output/` or `wandb/`; justify any large files and avoid polluting the package tree. Remove stale scratch runs only after results are logged.

Following these practices keeps the repository reproducible and ready for both human and AI contributors. Refer back to `docs/train.md`, `docs/eval_dataset.md`, and `docs/eval_mldr.md` whenever updating training, evaluation, or dataset pipelines.
