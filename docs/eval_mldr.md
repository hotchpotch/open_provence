# MLDR Evaluation Guide

This document explains how to regenerate MLDR pruning runs with `scripts/eval_mldr.py`. It focuses on **workflow and tooling** rather than reporting metrics, so you can reproduce results or adapt the pipeline for new checkpoints.

## 1. What is the MLDR dataset?

[`Shitao/MLDR`](https://huggingface.co/datasets/Shitao/MLDR) is a multilingual long-document retrieval benchmark. Each record contains:

- a query string,
- one positive passage that should remain after pruning,
- several negative passages that are expected to be removed,
- metadata such as language tags and document identifiers.

By default we evaluate `test[:200]`, but you can point the script at any MLDR slice via `--split` (e.g., `--split test --limit 500`, or `--split "dev[:100]"`). The positive + negative passages are long-form text; many exceed several kilobytes, so pruning is essential before running an LLM judge.

### Known data issues and ignore list

- Some queries (especially in Japanese) are malformed or not natural language questions.
- A handful of passages exceed 60k UTF-8 characters; letting them pass through causes GPU OOM or very slow CPU runs.
- We maintain a curated ignore list at `scripts/eval_mldr/ignored_questions.yaml`, keyed by language (`en`, `jp`). The CLI points to this file by default; if it is missing you can use `--force-no-ignore` to continue, but doing so is discouraged unless you understand the impact on metrics.
- If you discover new outliers (e.g., corrupted glyphs or language-mismatched queries), append them to the YAML file with a short comment describing the reason.

## 2. Prerequisites

1. **Dependencies** – install the project environment:
   ```bash
   uv sync
   ```
   On Linux x86_64 this resolves the CUDA 12.8 wheel by default; pass `--no-default-groups --group dev --group cpu`
   if you need a CPU/Metal-only setup.
2. **Credentials**
   - Hugging Face account (for dataset/model downloads, optional if cached).
   - OpenAI API key exported as `OPENAI_API_KEY` (LiteLLM forwards requests to `gpt-5-nano` by default).
3. **Hardware**
   - GPU is recommended but not required. The script automatically selects CUDA when available. CPU-only runs will succeed but take longer.
4. **Optional tooling**
   - For Naver Provence models (`naver/provence-*`, `naver/xprovence-*`), install spaCy sentence splitter dependencies referenced in the script comments if you plan to run those baselines.

## 3. Boilerplate command

The unified entry point is `scripts/eval_mldr.py`. It can run the pruning stage (`model.process`) and the LLM judging stage back-to-back, or you can execute them separately.

```bash
uv run python scripts/eval_mldr.py \
  --model output/release_models/open-provence-reranker-v1-gte-modernbert-base \
  --lang en \
  --threshold 0.1 \
  --limit 200 \
  --split test \
  --ignore-file scripts/eval_mldr/ignored_questions.yaml \
  --concurrency 5 \
  --output-dir output/eval_mldr_runs/open-provence_en_th_0_1 \
  --no-progress
```

We typically sweep four pruning thresholds—`0.05`, `0.1`, `0.3`, and `0.5`—to mirror the dataset benchmarks. You can launch consecutive runs with a simple loop:

```bash
for th in 0.05 0.1 0.3 0.5; do
  uv run python scripts/eval_mldr.py \
    --model output/release_models/open-provence-reranker-v1-gte-modernbert-base \
    --lang en \
    --threshold "$th" \
    --limit 200 \
    --split test \
    --ignore-file scripts/eval_mldr/ignored_questions.yaml \
    --concurrency 5 \
    --output-dir output/eval_mldr_runs/open-provence_en_th_${th//./_} \
    --no-progress
done
```

Key flags:

| Flag | Purpose |
| --- | --- |
| `--model` | Hugging Face ID or local checkpoint directory. Local paths are auto-resolved to `final_model/` when present. |
| `--lang` | Logical evaluation language (`en` or `jp`) used for the ignore list and reporting. |
| `--threshold` | Pruning threshold. Higher values prune more aggressively. |
| `--text-source` | Choose `pruned` (default) to evaluate the model outputs or `original` to judge raw passages. |
| `--limit` | Number of MLDR queries to sample (default `200`). Combine with `--split` for custom slices. |
| `--split` | Hugging Face split expression (`test`, `test[:200]`, etc.). Defaults to `test`. |
| `--ignore-file` | YAML with query IDs to skip; defaults to `scripts/eval_mldr/ignored_questions.yaml`. |
| `--output-dir` | Destination folder for artifacts. |

**Optional knobs** (use as needed):

- `--mldr-lang` — Override the MLDR dataset language when it differs from `--lang`.
- Sentence splitting is auto-detected (`auto`) and seamlessly handles Japanese and English without manual overrides.
- `--batch-size` — Controls batch size passed to `model.process` (default `16`). Lower it for tight GPU memory.
- `--device`, `--torch-dtype` — Pin execution to `cuda`/`cpu` or force precision (`float16`, `bfloat16`, etc.). When a Naver Provence Hub model is detected, the script auto-selects CUDA + `bfloat16` unless overridden.
- `--max-length` — Cap the token window supplied to the model loader.
- `--log-timing`, `--reranker-first-score`, `--no-progress` — Enable extra logging, change reranker aggregation, or silence progress bars.
- `--llm-model`, `--reasoning-effort` — Switch the LiteLLM backend and effort level for the judging stage (`gpt-5-nano` + `low` by default).
- `--force-process`, `--skip-process`, `--no-eval`, `--force-eval` — Control whether each stage runs when outputs already exist.
- `--include-negatives`, `--max-text-chars`, `--concurrency`, `--retries`, `--retry-delay`, `--request-timeout`, `--force-no-ignore` — Tune the evaluation filter, truncation length (default `60,000` characters), request parallelism, retry policy, and ignore-list usage.

### Output structure

```
output-dir/
  process/                # used when --text-source pruned (default)
    dataset/              # HF Dataset with pruned passages
    summary.json          # stats such as avg compression, runtime
    result.md             # sample inspection notebook
  process_original/       # populated when --text-source original
    dataset/
    summary.json
    result.md
  eval_llm/
    raw/                  # LiteLLM judgment cache (optional)
    summary.json          # aggregate LLM results
    summary.md            # human-readable recap
```

## 4. Running baselines and partial reruns (LLM evaluation on MLDR dataset)

- **Pruned run only**: If you just want to rebuild the pruned dataset without LLM judging, add `--no-eval`. You can judge later with `--skip-process --force-eval`.
- **Original text baseline**: Use `--text-source original` to bypass pruning and evaluate the raw passages. This is helpful for establishing upper bounds. Example:
  ```bash
  uv run python scripts/eval_mldr.py \
    --model output/release_models/open-provence-reranker-v1-gte-modernbert-base \
    --lang en \
    --text-source original \
    --limit 200 \
    --split test \
    --ignore-file scripts/eval_mldr/ignored_questions.yaml \
    --output-dir output/eval_mldr_runs/open-provence_en_original \
    --no-progress --force-process --force-eval
  ```
- **Reuse existing data**: When you just need to re-run the judge (e.g., switching LLM models), point to the same `output-dir` and pass `--skip-process --force-eval`.

## 5. Notes for Naver checkpoints

`naver/provence-reranker-debertav3-v1` and `naver/xprovence-reranker-bgem3-v1` rely on custom CUDA kernels. These may fail on very new GPU architectures. If you encounter NVRTC `--gpu-architecture` errors:

1. Retry on a machine with an Ampere/Lovelace GPU where CUDA 12.x kernels are tested.
2. Lower `--batch-size` (e.g., 2) and keep `--torch-dtype bfloat16` for Provence.
3. If GPU execution remains unstable, fall back to CPU (`--device cpu --torch-dtype float32`), understanding that the run will take significantly longer.

## 6. After the run

1. Copy `summary.{json,md}` to the release tree, e.g.,
   ```bash
   cp output/eval_mldr_runs/open-provence_en_th_0_1/eval_llm/summary.{json,md} \
      output/release_models/open-provence-reranker-v1-gte-modernbert-base/eval_results/mldr_en_th_0_1/eval_llm/
   ```
2. Archive the full `process/`（または `process_original/`）ディレクトリを `tmp/mldr_runs/<model>/<slug>/` に保管しておくと再判定が容易です。
3. Update documentation or reports (e.g., `docs/eval_reports/<date>.md`) only after verifying the aggregate JSON values.

## 7. Checklist

- [ ] Data split is `test[:200]` (or another documented limit).
- [ ] Ignore list points to `scripts/eval_mldr/ignored_questions.yaml`.
- [ ] `process/summary.json` shows realistic compression percentages (no unexpected 0%/100% unless running `original` mode).
- [ ] `eval_llm/summary.json` contains non-zero `records_evaluated` and `failed` is zero.
- [ ] Summaries copied into `output/release_models/<model>/eval_results/…`.

Following these steps ensures consistent MLDR pruning experiments across releases and team members. If you extend the pipeline—e.g., using a different judge model or language split—document the new flags and any additional ignore rules alongside the run artefacts.
