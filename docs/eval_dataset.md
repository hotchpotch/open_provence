# Dataset Evaluation Guide

`scripts/eval_datasets.py` measures how many annotated evidence spans survive pruning across a configuration of context-relevance datasets. This document explains how to run the CLI, what each configuration does, and how to interpret the generated artefacts. It intentionally omits score tables so the instructions remain evergreen.

## 1. What the script checks

For each dataset in a config file, the script:

1. Loads the dataset from Hugging Face (e.g., `hotchpotch/msmarco-context-relevance`).
2. Runs `model.process()` to prune each passage.
3. Compares the pruned spans against the labelled evidence annotations.
4. Computes span-level precision, recall, and a β = 2 F2 score (recall-weighted) plus mean compression.

Dropping relevant spans (false negatives) is more damaging than keeping surplus context, so F2 is the headline metric.

## 2. Known gotchas in the datasets

- Some datasets contain very long passages (>60 k characters). If you hit memory errors, temporarily limit evaluation via `--limit`, use the nano configs, or regenerate the dataset with shorter spans.
- A small number of queries in the multilingual sets are malformed or language-mismatched. The published configs already omit the worst offenders. If you uncover new issues, send a PR to update the source dataset rather than editing the evaluation script.
- Compression percentages are per-dataset averages; for heterogeneous corpora (e.g., GooAQ vs. JA-focused Wikipedia), expect different baseline compression even at the same threshold.

## 3. Config files

All configs live under `configs/eval_datasets/`:

| File | Purpose |
| --- | --- |
| `ja.yaml`, `en.yaml` | Full evaluation suites (all datasets, full sample counts). |
| `ja_nano.yaml`, `en_nano.yaml` | “Nano” subsets with per-dataset `n_samples` overrides for quick smoke tests. Use these when iterating on code or verifying regressions; they run 10–20× faster. |

Each entry in a config looks like:

```yaml
- dataset_name: hotchpotch/msmarco-context-relevance
  subset: default
  n_samples: 100        # only in *_nano.yaml
```

The script reads each row sequentially. You can clone a file and add/remove datasets for ad‑hoc scenarios. Optional keys:

- `split`: override the global split from the YAML header.
- `n_samples`: cap the number of records loaded from that dataset (only present in `*_nano.yaml`).

## 4. Core command template

```bash
uv run python scripts/eval_datasets.py \
  --config CONFIG_PATH \
  --model MODEL_DIR \
  --threshold 0.1 \
  --batch-size 256 \
  --timing-details \
  --output-json tmp/eval_<label>.json \
  --output-file tmp/eval_<label>.md
```

Replace:

- `CONFIG_PATH` with one of the YAML files described above.
- `MODEL_DIR` with the local run directory or exported checkpoint (the script auto-detects `final_model/` when present).
- `--threshold` with the pruning threshold you want to sweep.
- `--thresholds` / `--th` (optional) to evaluate multiple thresholds in one run.

Useful flags:

| Flag | Description |
| --- | --- |
| `--thresholds/--th` | Comma-separated list of extra thresholds. Repeat the flag to add more. |
| `--split` | Override the split for every dataset (default: YAML `split`). |
| `--limit` | Evaluate only the first *N* examples per dataset (applied after any `n_samples` cap). |
| `--target` | Restrict evaluation to specific datasets (`dataset_name:subset`). Repeatable. |
| `--device` | Force a specific device (e.g., `cuda`, `cuda:1`, `cpu`). |
| `--batch-size` | Controls the number of examples passed to `model.process` per call (default `512`). |
| `--no-progress` / `--silent` | Suppress progress bars or all intermediate logs. |
| `--timing-details` | Print per-stage timing and include them in the summaries. |
| `--use-automodel` | Load checkpoints via `transformers.AutoModel` (useful for remote-code models). |

The Markdown (`.md`) and JSON (`.json`) summaries land wherever you point `--output-file` / `--output-json`. Many workflows use `tmp/` during iteration, then copy artefacts under `output/release_models/<model>/eval_results/` when ready to publish.

### Nano quick-checks

To run a fast smoke test:

```bash
uv run python scripts/eval_datasets.py \
  --config configs/eval_datasets/ja_nano.yaml \
  --model output/release_models/open-provence-reranker-v1 \
  --threshold 0.1 \
  --batch-size 128 \
  --output-json tmp/eval_v1_ja_nano.json
```

The `*_nano.yaml` configs cap each dataset at `n_samples=100`, so the whole run finishes in a few minutes while still hitting every dataset.

## 5. Threshold sweeps

To sweep thresholds without editing the YAML, call the script in a loop:

```bash
for th in 0.05 0.1 0.3 0.5; do
  uv run python scripts/eval_datasets.py \
    --config configs/eval_datasets/en.yaml \
  --model output/release_models/open-provence-reranker-v1-gte-modernbert-base \
    --threshold "$th" \
    --batch-size 256 \
    --output-json tmp/eval_en_th_${th//./_}.json \
    --output-file tmp/eval_en_th_${th//./_}.md
done
```

Combine this with the nano configs for rapid iteration:

```bash
for th in 0.05 0.1; do
  uv run python scripts/eval_datasets.py \
    --config configs/eval_datasets/en_nano.yaml \
  --model output/release_models/open-provence-reranker-v1-gte-modernbert-base \
    --threshold "$th" \
    --batch-size 128 \
    --output-json tmp/eval_en_nano_th_${th//./_}.json
done
```

Once the numbers look good, re-run with the full configs to generate publishable artefacts.

## 6. After the run

1. Inspect `tmp/eval_<label>.json` and check that:
   - `macro` metrics (macro F2/recall/precision) are within expected ranges.
   - `datasets` entries are present for every config row.
2. Copy the JSON and Markdown into the release tree, e.g.,
   ```bash
   cp tmp/eval_en_th_0_1.{json,md} \
      output/release_models/open-provence-reranker-v1-gte-modernbert-base/eval_results/eval_datasets_en.*
   ```
3. Update any dated reports (e.g., `docs/eval_reports/<date>.md`) if the numbers correspond to a release milestone.

## 7. Checklist

- [ ] Config matches the intended language (no `en` model on `ja.yaml` without overrides).
- [ ] Ignore file includes newly discovered bad records when necessary.
- [ ] Markdown/JSON summaries stored under `tmp/` have been archived or copied to the release directory.
- [ ] Threshold sweeps include both full and nano configs during development.

Following this playbook keeps dataset-level pruning evaluations reproducible and manageable, whether you are iterating on new pruning heads or validating regression fixes.
