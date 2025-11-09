# Training Workflow

This guide walks through the end-to-end process for training OpenProvence reranker–pruner models. It introduces the ready-made configs, explains the YAML structure, and highlights what to look for when validating a fresh run.

## Prerequisites
- Run `uv sync` to install the base environment. On Linux x86_64 this now resolves the CUDA 12.8 wheel (`torch==2.8.0+cu128`); pass `--no-default-groups --group dev --group cpu` if you need the CPU/Metal wheel instead.
- Sign in to the Hugging Face Hub if any referenced datasets require authentication.
- Export `WANDB_API_KEY` when you want metrics in the shared Weights & Biases project.
- Use a single NVIDIA GPU with ≥16 GB of memory; every recipe in this guide fits that footprint. **Installing FlashAttention (`uv sync --group flash-attn` or by adding the vetted wheel in `tmp/`) delivers a noticeable speed-up.**

## Ready-to-Use Configurations
- Release checkpoints (all timed on an RTX 5090)  
  - [configs/open-provence-reranker-v1-gte-modernbert-base.yaml](../configs/open-provence-reranker-v1-gte-modernbert-base.yaml) — English-only ModernBERT backbone trained on MSMARCO/NQ/GooAQ. **≈10 hours**.
  - [configs/open-provence-reranker-v1.yaml](../configs/open-provence-reranker-v1.yaml) — Japanese & English dual-language recipe backed by `hotchpotch/japanese-reranker-base-v2` and the full multilingual corpus. **≈10 hours**.
  - [configs/open-provence-reranker-xsmall-v1.yaml](../configs/open-provence-reranker-xsmall-v1.yaml) — 30M-parameter bilingual checkpoint optimised for latency-sensitive deployments. **≈5 hours**.
  - [configs/open-provence-reranker-large-v1.yaml](../configs/open-provence-reranker-large-v1.yaml) — 310M-parameter bilingual checkpoint tuned for maximum retention and compression. **≈20 hours**.
- Toy pipelines (smoke tests, 5–10 minutes on an RTX 5090)  
  - [configs/toy-open-provence-reranker-v1-gte-modernbert-base.yaml](../configs/toy-open-provence-reranker-v1-gte-modernbert-base.yaml) — English toy run sampling ≈12 k examples.
  - [configs/toy-open-provence-reranker-v1.yaml](../configs/toy-open-provence-reranker-v1.yaml) — Japanese & English toy run mirroring the release mixture at tiny scale.

Clone one of these files for experiments and keep your custom versions under `configs/`.

## Running a Training Job

### Quick toy run (5–10 minutes)

Before you launch the full recipes, warm up the pipeline with the toy configs. They sample a tiny slice of each dataset, train in roughly 5–10 minutes on an RTX 5090, and still produce sensible pruning behaviour.

```bash
# English toy model
uv run open_provence_trainer configs/toy-open-provence-reranker-v1-gte-modernbert-base.yaml

# Japanese & English toy model
uv run open_provence_trainer configs/toy-open-provence-reranker-v1.yaml
```

If the logs or nano metrics drift far from the reference values in the “Toy Dataset Reference” section below, double-check your environment before advancing to the full-scale runs.

### English full run
```bash
uv run open_provence_trainer configs/open-provence-reranker-v1-gte-modernbert-base.yaml
```

The trainer prints the parsed arguments, then begins a single epoch with `per_device_train_batch_size=4` and `gradient_accumulation_steps=64` (effective batch size 256). Logs appear every 100 steps, and evaluation runs every 500 steps.

### Japanese & English full run
```bash
uv run open_provence_trainer configs/open-provence-reranker-v1.yaml
```

This configuration processes a larger corpus, so expect longer wall-clock time. Several datasets include `upsample_factor` to balance coverage across domains.

### Monitoring and artifacts
- Outputs live under `output/<config>_<timestamp>/`. The final checkpoint is always stored in `final_model/`.
- When `report_to=["wandb"]`, runs are uploaded to the `hotchpotch/open-provence` project with slug `<config>-<timestamp>`.
- After training, the `eval_datasets` block automatically kicks off `scripts/eval_datasets.py` with the language-appropriate config so you get nano evaluation results without additional commands.

### Resuming after an interruption

Interrupted runs keep Hugging Face–style checkpoints inside the training output directory (`checkpoint-3000/`, `checkpoint-3500/`, ...). Restart training with any of the following options:

- Command line: `uv run open_provence_trainer <config.yaml> --checkpoint /path/to/output/run_dir` automatically resumes from the latest `checkpoint-*` under that directory. To pin a given step, pass the checkpoint directory itself (e.g., `--checkpoint /.../checkpoint-5000`).
- Hugging Face style: `--resume_from_checkpoint /.../checkpoint-5000` (or the YAML equivalent `training_args.resume_from_checkpoint`) also works; we still auto-set `output_dir` to the checkpoint’s parent run directory so artifacts stay together.
- Config-driven: add `training_args.checkpoint: /.../output/run_dir` (parent) or `training_args.resume_from_checkpoint: /.../output/run_dir/checkpoint-5000` when you want the recipe to resume automatically.

The trainer validates that every resolved checkpoint contains `trainer_state.json` and prints which directory it picked (including the step number) before restarting so you can verify the resume target.

## Configuration Anatomy

Every config has the same high-level shape:

```yaml
model_args:
  model_name_or_path: ...
  classifier_dropout: ...

data_args:
  datasets:
    - dataset_name: ...
      subset: ...
      teacher_column: ...
      items: ...           # optional: limit contexts per query
      n_samples: ...       # optional: sample cap (toy configs)
      upsample_factor: ... # optional: repeat dataset during sampling

training_args:
  learning_rate: ...
  per_device_train_batch_size: ...
  gradient_accumulation_steps: ...
  num_train_epochs: 1
  bf16: true
  dataloader_num_workers: 8
  eval_steps: 500
  report_to: ["wandb"]
  eval_datasets:
    config: ...
    threshold: 0.1
    batch_size: 32
```

### Important fields
- **`model_args`** — selects the base encoder. Swap `model_name_or_path` when trying a new backbone.
- **`data_args.datasets`** — controls dataset mixing. Useful options:
  - `items`: how many passages to sample per query (acts like negative sampling).
  - `n_samples`: cap dataset size for faster iteration (used in toy configs).
  - `upsample_factor`: repeat a dataset to boost its contribution.
  - `teacher_column`: column containing teacher reranker scores for distillation.
- **`training_args`** — mirrors Hugging Face `TrainingArguments` with pruning defaults:
  - Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps`.
  - Mixed precision uses BF16 by default; enable `fp16` if your hardware lacks BF16 support.
  - `eval_steps` and `logging_steps` dictate evaluation cadence and logging granularity.
  - The nested `eval_datasets` block defines which evaluation YAML runs automatically when training completes.

When customising, clone the closest template, tweak the dataset list, and adjust batch size or accumulation to fit your GPU memory budget.

## After Training
1. Inspect `output/<config>_<timestamp>/final_model/` to confirm weights, tokenizer files, and evaluation summaries were produced.
2. Copy nano evaluation artifacts into your release folder or follow up with the full evaluation suites described in `docs/eval_dataset.md` and `docs/eval_mldr.md`.
3. Record the exact command, config, and output path in your PR or experiment log. This keeps reproducibility tight across the team.

## Toy Dataset Reference (2025-10-29)

Run the toy config whenever you need a quick health check:

```bash
uv run open_provence_trainer configs/toy-open-provence-reranker-v1-gte-modernbert-base.yaml
```

On an RTX 5090 this finishes in ~5 minutes (effective batch size 64). Expect logs similar to:

```
{'eval_loss': 0.2196, 'eval_pruning_loss': 0.2269, 'eval_ranking_loss': 0.0560, 'step': 180}
{'loss': 0.4167, 'pruning_loss': 0.4147, 'ranking_loss': 0.0386, 'step': 181}
Training completed successfully!
Model saved to: ./output/toy-open-provence-reranker-v1-gte-modernbert-base_20251029_090143/final_model
```

Automatic nano evaluation metrics at threshold 0.1:

| Dataset | F2 | Recall | Precision | Mean Compression (%) | Span Accuracy |
| --- | --- | --- | --- | --- | --- |
| hotchpotch/msmarco-context-relevance:freq2 | 0.7887 | 0.9947 | 0.4314 | 10.41 | 0.4919 |
| hotchpotch/natural-questions-context-relevance:nodup_freq2 | 0.6513 | 0.9013 | 0.3088 | 73.80 | 0.8363 |
| hotchpotch/gooaq-context-relevance-130k:default | 0.8214 | 0.9782 | 0.5006 | 46.65 | 0.7208 |

### Japanese & English toy run (2025-10-30)

```bash
uv run open_provence_trainer configs/toy-open-provence-reranker-v1.yaml
```

The run completed in ~4 minutes 22 seconds on an RTX 5090 (effective batch size 64). Artefacts live under
`output/toy-open-provence-reranker-v1_20251030_092420/final_model`, and the automatic nano evaluation used
`configs/eval_datasets/ja_nano.yaml` at threshold 0.1. Expect evaluation logs similar to:

| Dataset | F2 | Recall | Precision | Mean Compression (%) | Span Accuracy |
| --- | --- | --- | --- | --- | --- |
| hotchpotch/msmarco-context-relevance:freq2 | 0.7831 | 0.9930 | 0.4243 | 9.65 | 0.4773 |
| hotchpotch/natural-questions-context-relevance:nodup_freq2 | 0.6036 | 0.9821 | 0.2375 | 62.54 | 0.7548 |
| hotchpotch/gooaq-context-relevance-130k:default | 0.7499 | 0.9716 | 0.3921 | 32.08 | 0.5706 |
| hotchpotch/japanese-context-relevance:msmarco-ja-freq2 | 0.8489 | 0.9664 | 0.5712 | 25.03 | 0.6970 |
| hotchpotch/japanese-context-relevance:auto-wiki-qa-nemotron | 0.7046 | 0.8835 | 0.3893 | 70.63 | 0.8727 |
| hotchpotch/japanese-context-relevance:jaquad-freq2 | 0.7152 | 0.9221 | 0.3770 | 72.37 | 0.8818 |
| hotchpotch/japanese-context-relevance:jqara | 0.6359 | 0.8279 | 0.3299 | 67.18 | 0.8376 |
| hotchpotch/japanese-context-relevance:jsquad-freq2 | 0.7280 | 0.8859 | 0.4250 | 62.27 | 0.8264 |
| hotchpotch/japanese-context-relevance:miracl | 0.8221 | 0.9529 | 0.5307 | 43.04 | 0.7808 |
| hotchpotch/japanese-context-relevance:mkqa | 0.6406 | 0.8682 | 0.3127 | 67.61 | 0.8437 |
| hotchpotch/japanese-context-relevance:mr-tydi | 0.8222 | 0.9508 | 0.5336 | 44.52 | 0.7919 |
| hotchpotch/japanese-context-relevance:quiz-no-mori | 0.6456 | 0.7701 | 0.3920 | 72.99 | 0.8651 |
| hotchpotch/japanese-context-relevance:quiz-works | 0.6516 | 0.8069 | 0.3683 | 70.80 | 0.8547 |
| hotchpotch/japanese-context-relevance:JFWIR | 0.6515 | 0.7901 | 0.3829 | 60.83 | 0.7377 |

These values illustrate that the Japanese slices stay within the expected F2 range (0.64–0.85) and compression
rates (25–73%) for healthy pruning behaviour. Investigate large deviations before proceeding to full-scale
runs.

**If your toy results diverge dramatically from these numbers, something is misconfigured—double-check dataset access, GPU precision settings, and the trainer logs before trusting the run.**
