# Training Workflow

This guide walks through the end-to-end process for training OpenProvence reranker–pruner models. It introduces the ready-made configs, explains the YAML structure, and highlights what to look for when validating a fresh run.

## Prerequisites
- Run `uv sync` to install the base environment (PyTorch 2.7.1 CPU/Metal build).
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

### Monitoring and artefacts
- Outputs live under `output/<config>_<timestamp>/`. The final checkpoint is always stored in `final_model/`.
- When `report_to=["wandb"]`, runs are uploaded to the `hotchpotch/open-provence` project with slug `<config>-<timestamp>`.
- After training, the `eval_datasets` block automatically kicks off `scripts/eval_datasets.py` with the language-appropriate config so you get nano evaluation results without additional commands.

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
2. Copy nano evaluation artefacts into your release folder or follow up with the full evaluation suites described in `docs/eval_dataset.md` and `docs/eval_mldr.md`.
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

The Japanese & English toy configuration (`configs/toy-open-provence-reranker-v1.yaml`) provides the same sanity check across the multilingual datasets with comparable F2 scores.

**If your toy results diverge dramatically from these numbers, something is misconfigured—double-check dataset access, GPU precision settings, and the trainer logs before trusting the run.**
