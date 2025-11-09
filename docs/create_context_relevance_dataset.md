# Building Context-Relevance Datasets

This guide describes how to convert sentence-transformer style triplet datasets into the training
format used by Open Provence models (`id`, `query`, `texts`, `context_spans`, `labels`). It covers
prerequisites, the conversion workflow, and how to append pruner-derived relevance labels and
cross-encoder teacher scores.

> **Note**: Any reranker that can be loaded as a [`SentenceTransformer`](https://www.sbert.net/) `CrossEncoder`
> is supported by the teacher-score augmentation script. You can mix English and Japanese models in the
> same dataset as long as the model tokenizer matches the language of the examples.

---

## 1. Dataset structure

Each converted dataset is saved with `datasets.Dataset.save_to_disk` and contains the following columns:

| Column | Description |
| --- | --- |
| `id` | Unique document identifier (string). |
| `query` | Query text (string). |
| `texts` | List of passage strings; index order matches `context_spans`. |
| `context_spans` | List of `[start, end)` character offsets into each passage; index order matches `texts`. |
| `labels` | Per-text binary labels (1 if the passage contains the positive span, otherwise 0). |
| `context_spans_relevance` | Binary span flags from the pruner (`1` keeps the span, `0` masks it). Treat as optional until you run the relevance augmentation step. |
| `teacher_scores.<name>` | Cross-encoder scores appended later; one column per reranker (optional). |

> **How training uses `labels`**
>
> - The column is *not* strictly required when the dataset ships with teacher scores. In that case the trainer consumes the `teacher_scores.*` columns for the regression objective and can fall back to deterministic zero labels.
> - When teacher scores are absent, the ranking head relies on `labels` as binary targets, so the column becomes mandatory for those configurations.
> - Even when teacher scores exist, `labels` are still read by the data-preparation pipeline (notably in `sample_items_by_label_priority`) to keep at least one positive candidate per query during the optional `items` down-sampling step. If you omit `labels`, that sampler degrades to uniform selection and may drop all positives in edge cases.

Datasets are split into `train`, `validation`, and `test`. If the source dataset is missing
`validation`/`test`, the converter samples 1% (up to 5k rows) from the shuffled `train` split to
create them; otherwise the upstream splits are preserved as-is.

---

## 2. Prerequisites

```bash
uv sync                              # installs CUDA-enabled deps on Linux x86_64
huggingface-cli login                # required if the source dataset is private
uv run pip install nltk fast-bunkai  # sentence splitters (punkt for en, bunkai for ja)
uv pip install vllm                  # optional, only needed for LLM helpers
uv run python -m nltk.downloader punkt  # required for --lang en

> Tip: create a dedicated vLLM environment with `uv venv path/to/other-venv-path`
> and install vLLM inside it. Direct `uv pip install vllm` can fail to resolve
> GPU wheels, while the separate pruning environment keeps those dependencies
> isolated from the main project tooling.
```

- CPU / Metal hosts: use `uv sync --no-default-groups --group dev --group cpu` if you cannot install CUDA wheels.
- English (`--lang en`) requires NLTK punkt.
- Japanese (`--lang ja`) requires `fast-bunkai`.
- The converter fails fast if the required splitter is not available.

---

## 3. Conversion workflow

The main entry point lives at `scripts/context-relevance-datasets/generate_ds_from_sentense_transformer.py`.
It consumes triplet datasets with `(query, positive, negative)` fields and produces the Provence format.

### CLI flags

| Flag | Purpose |
| --- | --- |
| `--dataset` | Hugging Face dataset ID (e.g., `tomaarsen/natural-questions-hard-negatives`). |
| `--subset` | Optional subset name (e.g., `triplet-5`). |
| `--lang` | Sentence splitter language (`en`, `ja`, or any punkt-supported locale code). |
| `--sample-size` | Optional sample size per split (shuffled with `--seed`). |
| `--output-root` | Base output directory (default `output/context-relevance-datasets/base`). |
| `--seed` | Shuffle seed (default `42`). |
| `--overwrite` | Allow overwriting an existing directory. |
| `--log-level` | Control logging verbosity (`DEBUG`/`INFO`/`WARNING`/`ERROR`). |

### Example (full Natural Questions)

```bash
uv run python scripts/context-relevance-datasets/generate_ds_from_sentense_transformer.py \
  --dataset tomaarsen/natural-questions-hard-negatives \
  --subset triplet-5 \
  --lang en \
  --overwrite
```

Result path: `output/context-relevance-datasets/base/tomaarsen_natural-questions-hard-negatives_triplet-5`

### Example (400k GooAQ sample)

```bash
uv run python scripts/context-relevance-datasets/generate_ds_from_sentense_transformer.py \
  --dataset tomaarsen/gooaq-hard-negatives \
  --subset triplet-5 \
  --lang en \
  --sample-size 400000 \
  --overwrite
```

### Example (Japanese triples)

```bash
uv run python scripts/context-relevance-datasets/generate_ds_from_sentense_transformer.py \
  --dataset hotchpotch/japanese-context-relevance \
  --subset msmarco-ja \
  --lang ja \
  --overwrite
```

### Recommended duplicate filtering

Before adding LLM annotations or reranker scores, run a text-level deduplication pass so that the `texts`
column contains a balanced mix of passages. Use
`scripts/context-relevance-datasets/frequency_filter_ds.py` to hash each passage, track how often it
appears, and remove rows whose duplicate count exceeds a threshold `N`. Keeping the post-filter duplicate
ratio in the `0–15%` range has produced more stable pruner training runs in practice—datasets with extreme
duplication push the model toward “keep everything.” For example, the raw
`hotchpotch/natural-questions-context-relevance` split started at ~86% duplicate passages; by exporting
filtered variants with `--threshold 0`, `1`, or `2`, the ratio dropped into the recommended zone while
leaving enough coverage for training. If your upstream pipeline synthesizes hard negatives, consider
deduplicating during data generation as well, but the CLI provides a fast safety net before scoring.

---

## 4. Verifying the converted dataset

```bash
uv run python - <<'PY'
from datasets import load_from_disk
from pprint import pprint

ds = load_from_disk("output/context-relevance-datasets/base/tomaarsen_natural-questions-hard-negatives_triplet-5")
print(ds)
for split in ["train", "validation", "test"]:
    print(split, len(ds[split]))
example = ds["train"][0]
pprint({k: example[k] for k in ["id", "query", "texts", "context_spans", "labels"]})
PY
```

### Sample preview from the released datasets

The snippet below loads the first ten records from the public datasets and renders each span using its
`context_spans` offsets together with the pruner relevance scores.

```bash
uv run python - <<'PY'
from datasets import load_dataset
from textwrap import shorten

MAX_RECORDS = 10
MAX_SPANS = 3

print("# English (MS MARCO)")
for record in load_dataset("hotchpotch/msmarco-context-relevance", split=f"train[:{MAX_RECORDS}]"):
    print(record["id"])
    for text_index, (text, spans, scores) in enumerate(zip(record["texts"], record["context_spans"], record["context_spans_relevance"])):
        for span_index, (start, end) in enumerate(spans[:MAX_SPANS]):
            snippet = shorten(text[start:end].replace("\n", " "), width=60, placeholder="...")
            relevance = scores[span_index] if span_index < len(scores) else 0
            print(f"- [{record['id']}][text:{text_index}, span:{span_index}, offsets:{start}-{end}, relevance:{relevance}] {snippet}")
    print()

print("# Japanese (MS MARCO JA)")
for record in load_dataset("hotchpotch/japanese-context-relevance", "msmarco-ja", split=f"train[:{MAX_RECORDS}]"):
    print(record["id"])
    for text_index, (text, spans, scores) in enumerate(zip(record["texts"], record["context_spans"], record["context_spans_relevance"])):
        for span_index, (start, end) in enumerate(spans[:MAX_SPANS]):
            snippet = shorten(text[start:end].replace("\n", " "), width=60, placeholder="...")
            relevance = scores[span_index] if span_index < len(scores) else 0
            print(f"- [{record['id']}][text:{text_index}, span:{span_index}, offsets:{start}-{end}, relevance:{relevance}] {snippet}")
    print()
PY
```

Example output (truncated):

```
# English (MS MARCO)
msmarco:95508
- [msmarco:95508][text:0, span:0, offsets:0-775, relevance:1] aud usd analysis outlook and forecasts aud usd australian...
- [msmarco:95508][text:1, span:0, offsets:0-360, relevance:1] the average rate of aud usd 0 71 for november 2015...
- [msmarco:95508][text:2, span:0, offsets:0-275, relevance:1] forecast chart com is forecasting that the exchange rate...

doc2...

# Japanese (MS MARCO JA)
msmarco-ja:95508
- [msmarco-ja:95508][text:0, span:0, offsets:0-74, relevance:0] ...
- [msmarco-ja:95508][text:1, span:0, offsets:0-87, relevance:1] ...
- [msmarco-ja:95508][text:2, span:0, offsets:0-76, relevance:1] Forecast Chart...
```

---

## 5. Adding pruner relevance labels

After conversion, attach pruner-derived span flags to each passage with
`add_context_spans_relevance.py`. The script streams queries/chunks through a vLLM endpoint, caches
intermediate batches on disk, and writes an augmented dataset alongside the original directory.

```bash
uv run python scripts/context-relevance-datasets/add_context_spans_relevance.py \
  --dataset-path output/context-relevance-datasets/base/tomaarsen_natural-questions-hard-negatives_triplet-5 \
  --model hotchpotch/query-context-pruner-multilingual-Qwen3-4B \
  --overwrite
```

### Key flags

| Flag | Purpose |
| --- | --- |
| `--dataset-path` | Source directory produced by the converter script. |
| `--dataset-name` | Optional cache key override; defaults to the dataset directory name. |
| `--model` | vLLM-compatible pruner checkpoint. |
| `--max-model-len` | Context window passed to vLLM (default `1280`). |
| `--batch-size` / `--group-size` | Control prompt batch sizes and cache shard sizes. |
| `--cache-root` | Cache directory (default `./cache/context_spans_relevance`). |
| `--force-reprocess` | Ignore cache hits and re-run inference. |
| `--debug` / `--debug-limit` | Process only the first *N* rows per split and write to `_with_relevance_debug`. |
| `--gpu-memory-utilization` | Fraction of GPU memory assigned to vLLM (default `0.9`). |
| `--temperature` / `--top-p` | Sampling parameters for the pruner LLM. |
| `--stop-tokens` | Stop sequence(s) for generation. |
| `--output-path` | Override the default `<dataset>_with_relevance` destination. |
| `--verbose` | Enable verbose logging. |
| `--overwrite` | Replace an existing output directory. |

The pruner currently emits binary `context_spans_relevance` lists. Each inner list aligns with
`context_spans`: `1` marks a span as relevant, `0` means the pruner would drop it. You can treat
these as soft labels or convert them to weights in downstream code.

Inspect the augmented result:

```bash
uv run python - <<'PY'
from datasets import load_from_disk

ds = load_from_disk(
    "output/context-relevance-datasets/base/"
    "tomaarsen_natural-questions-hard-negatives_triplet-5_with_relevance"
)
print(ds)
example = ds["train"][0]
print("span flags:", example["context_spans_relevance"][0][:5])
PY
```

## 6. Adding reranker teacher scores

Use `add_reranker_teacher_scores.py` to append `teacher_scores.<column-name>` columns for each
cross-encoder. You can re-run the script with different models to stack multiple columns in the same
dataset directory.

### Example commands

```bash
uv run python scripts/context-relevance-datasets/add_reranker_teacher_scores.py \
  --dataset-path output/context-relevance-datasets/base/tomaarsen_natural-questions-hard-negatives_triplet-5_with_relevance \
  --model Alibaba-NLP/gte-reranker-modernbert-base \
  --column-name gte-modernbert-base \
  --overwrite

uv run python scripts/context-relevance-datasets/add_reranker_teacher_scores.py \
  --dataset-path output/context-relevance-datasets/base/hotchpotch_japanese-context-relevance_msmarco-ja_with_relevance \
  --model hotchpotch/japanese-reranker-xsmall-v2 \
  --column-name japanese-reranker-xsmall-v2 \
  --overwrite
```

### Key flags

| Flag | Purpose |
| --- | --- |
| `--dataset-path` | Input directory (typically the `_with_relevance` output). |
| `--output-path` | Optional destination; defaults to appending `_with_teacher_scores`. |
| `--model` | Cross-encoder checkpoint ID. |
| `--column-name` | Suffix for the new column; defaults to the sanitized model name. |
| `--batch-size` | Number of query/passage pairs per forward call (default `16`). |
| `--dtype` | Force model dtype (`float32`, `float16`, `bfloat16`). |
| `--debug-limit` | Truncate each split for smoke tests. |
| `--validate-samples` | Print the first *N* rows for sanity checking. |
| `--log-level` | Set logging verbosity for the run. |
| `--overwrite` | Replace an existing output path. |

Check the resulting scores:

```bash
uv run python - <<'PY'
from datasets import load_from_disk

ds = load_from_disk(
    "output/context-relevance-datasets/base/"
    "tomaarsen_natural-questions-hard-negatives_triplet-5_with_relevance_with_teacher_scores"
)
example = ds["train"][0]
for key in sorted(k for k in example if k.startswith("teacher_scores.")):
    print(key, example[key][:3])
PY
```

## 7. Uploading to the Hugging Face Hub

Finalize the pipeline by pushing the directory to the Hub with
`upload_context_relevance_to_hf.py`.

```bash
uv run python scripts/context-relevance-datasets/upload_context_relevance_to_hf.py \
  --dataset-path output/context-relevance-datasets/base/tomaarsen_natural-questions-hard-negatives_triplet-5_with_relevance_with_teacher_scores \
  --repo-id hotchpotch/natural-questions-context-relevance \
  --subset default \
  --commit-message "Add NQ context relevance with teacher scores"
```

### Useful flags

| Flag | Purpose |
| --- | --- |
| `--split` | Push a single split instead of the full DatasetDict. |
| `--max-shard-size` / `--num-shards` | Tune Parquet sharding. |
| `--num-proc` | Parallelism for pre-upload preparation. |
| `--commit-description` / `--revision` | Extra metadata for the Hub commit. |
| `--token` | Explicit Hugging Face access token (defaults to logged-in user). |
| `--public` | Publish the repo (private by default). |
| `--no-embed` | Disable embedding of external files inside the shards. |
| `--dry-run` | Print parameters without uploading (great for verification). |

The helper prints row counts and estimated size before uploading. When `--dry-run` is set, you can
review the parameters without creating a Hub commit.

## 8. Released deduplicated subsets (2025-10-26)

The table below summarises the English and Japanese context-relevance datasets after applying
`frequency_filter_ds.py`. Row counts refer to the train split; validation/test remain unchanged.
Duplicate ratios measure the share of duplicate `texts` entries (lower is better). ✅ marks the subset
currently uploaded to the Hugging Face Hub.

| Dataset | Original train rows | Original duplicate texts | Original dup ratio | freq0 rows<br/>dup ratio | freq1 rows<br/>dup ratio | freq2 rows<br/>dup ratio | Published subset |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| hotchpotch/msmarco-context-relevance | 492,729 | 1,533,936 | 38.91% | 152,841<br/>0.00% | 212,517<br/>5.16% | 262,436<br/>10.35% | ✅ freq2 |
| hotchpotch/natural-questions-context-relevance | 94,734 | 493,445 | 86.81% | 5,527<br/>4.03% | 8,437<br/>14.43% | 11,873<br/>25.53% | ✅ freq1 |
| hotchpotch/gooaq-context-relevance-400k | 392,040 | 1,523,017 | 64.75% | 56,675<br/>10.07% | 89,143<br/>17.14% | 127,468<br/>25.85% | ✅ freq0 |
| hotchpotch/japanese-context-relevance (msmarco-ja) | 492,729 | 1,533,961 | 38.91% | 152,841<br/>0.00% | 212,520<br/>5.16% | 262,438<br/>10.35% | ✅ msmarco-ja-freq2 |
| hotchpotch/japanese-context-relevance (jsquad) | 55,065 | 188,472 | 42.78% | 10,429<br/>0.00% | 19,253<br/>7.85% | 26,523<br/>14.46% | ✅ jsquad-freq2 |
| hotchpotch/japanese-context-relevance (jaquad) | 25,329 | 77,076 | 38.04% | 5,523<br/>0.00% | 10,609<br/>7.73% | 14,050<br/>13.50% | ✅ jaquad-freq2 |

Mirror the published configuration by pointing training configs to the appropriate subset (see the
`freq-open-provence-reranker-*.yaml` examples) before launching new runs or exporting Hub-ready models.

## 9. End-to-end checklist

1. Convert the triplet dataset with `generate_ds_from_sentense_transformer.py`.
2. Spot-check the base DatasetDict (row counts, span offsets, labels).
3. Run `add_context_spans_relevance.py` to populate `context_spans_relevance` flags (or its debug
   mode during development).
4. Optionally append one or more `teacher_scores.*` columns with
   `add_reranker_teacher_scores.py`.
5. Upload the final directory to the Hub—or point Provence training jobs at the local path.

Each stage writes to a new directory, so you can resume or branch off any step without mutating the
previous artefacts.
