# OpenProvence: Open-Source, Efficient, and Robust Context Pruning for Retrieval-Augmented Generation

> ✂️ Lightweight Provence-style rerankers that keep the answers and drop the noise for retrieval-augmented generation.

<p align="left">
  <a href="https://huggingface.co/spaces/hotchpotch/open_provence_demo">🤗 Spaces WebUI Demo</a> ·
  <a href="docs/eval_reports/open_provence_v1_eval_report.md">Evaluation Report</a> ·
  <a href="docs/train.md">Training Guide</a> ·
  <a href="docs/create_context_relevance_dataset.md">Dataset Pipeline</a>
</p>

OpenProvence is a fully open-source implementation of Provence-style reranker–pruner models. These models remove irrelevant context while simultaneously assigning relevance scores, following the approach introduced in the [Provence paper](https://arxiv.org/abs/2501.16214). Modern agentic workflows—DeepResearch loops, context-engineering pipelines, autonomous search agents—tend to accumulate large amounts of tangential context. Drop a lightweight Provence model in front of your LLM to trim token spend and keep the passages that actually answer the query.

## ✨ Highlights

- **Pruning power** – Drop ~99% of off-topic sentences while still compressing 80–90% of relevant text; MLDR evaluations confirm the answers stay intact.
- **Model zoo you can ship today** – Four checkpoints (30M–310M parameters) covering English and Japanese, each published on Hugging Face under the MIT License. The 30M xsmall model runs comfortably on CPU and absolutely flies on GPU.
- **Reproducible training** – Follow the playbook in [docs/train.md](docs/train.md) to train every checkpoint on a single ≥16 GB NVIDIA GPU.
- **Dataset tooling** – Pipelines for building OpenProvence-format corpora from your own data, documented in [docs/create_context_relevance_dataset.md](docs/create_context_relevance_dataset.md).
- **Evaluation stack** – CLI utilities for dataset-retention sweeps and MLDR long-document runs ([docs/eval_dataset.md](docs/eval_dataset.md) / [docs/eval_mldr.md](docs/eval_mldr.md)).
- **Observability built-in** – Consolidated metrics, plots, and commentary live in the [OpenProvence v1 Evaluation Report](docs/eval_reports/open_provence_v1_eval_report.md).
- **Teacher model** – Ship your own labels with the multilingual span annotator [query-context-pruner-multilingual-Qwen3-4B](https://huggingface.co/hotchpotch/query-context-pruner-multilingual-Qwen3-4B).

## 📦 Model Catalog

Pick the checkpoint that matches your latency and language requirements. All weights are hosted on Hugging Face with permissive licensing.

| Short Name | Language | Backbone | Hugging Face Model ID | Parameters | Notes |
|------------|----------|----------|------------------------|------------|-------|
| base | English, Japanese | ModernBERT | [hotchpotch/open-provence-reranker-v1](https://huggingface.co/hotchpotch/open-provence-reranker-v1) | 130M | Balanced accuracy vs. speed for English + Japanese |
| xsmall | English, Japanese | ModernBERT | [hotchpotch/open-provence-reranker-xsmall-v1](https://huggingface.co/hotchpotch/open-provence-reranker-xsmall-v1) | 30M | Fastest checkpoint; keeps MLDR scores with modest pruning at th=0.05 |
| large | English, Japanese | ModernBERT | [hotchpotch/open-provence-reranker-large-v1](https://huggingface.co/hotchpotch/open-provence-reranker-large-v1) | 310M | Highest compression at similar F2, best when latency budget allows |
| en-gte | English | ModernBERT | [hotchpotch/open-provence-reranker-v1-gte-modernbert-base](https://huggingface.co/hotchpotch/open-provence-reranker-v1-gte-modernbert-base) | 149M | English-only; likely the strongest English reranker score-wise |


## 🚀 Quickstart

### 🖥️ Web App (Gradio)

Try the hosted [OpenProvence inference demo on 🤗 Spaces](https://huggingface.co/spaces/hotchpotch/open_provence_demo). Run the same interface locally with:

```bash
git clone https://huggingface.co/spaces/hotchpotch/open_provence_demo
cd open_provence_demo
uv sync
uv run python app.py
```


### 🐍 Python API

```python
from transformers import AutoModel

model_name = "hotchpotch/open-provence-reranker-xsmall-v1"
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
)

question:str = "What's your favorite Japanese food?"
context:str = """
Work deadlines piled up today, and I kept rambling about budget spreadsheets to my roommate.
Next spring I'm planning a trip to Japan so I can wander Kyoto's markets and taste every regional dish I find.
Sushi is honestly my favourite—I want to grab a counter seat and let the chef serve endless nigiri until I'm smiling through soy sauce.
Later I remembered to water the plants and pay the electricity bill before finally getting some sleep.
"""

result = model.process(
    question=question,
    context=context,
    threshold=0.1,
    show_progress=True,
)

print("Pruned context:\\n" + result["pruned_context"])
print("Reranking score:", round(result["reranking_score"], 4))
print("Compression rate:", round(result["compression_rate"], 2))
```

```
Pruned context:
Next spring I'm planning a trip to Japan so I can wander Kyoto's markets and taste every regional dish I find.
Sushi is honestly my favourite—I want to grab a counter seat and let the chef serve endless nigiri until I'm smiling through soy sauce.

Reranking score: 0.6448
Compression rate: 44.37
```

Passing a single string returns the pruned text plus scalar scores. Switch `model_name` to `"hotchpotch/open-provence-reranker-v1-gte-modernbert-base"` for the English checkpoint or any other published run. In practice you can batch hundreds of question–context pairs at once to maximise GPU throughput.

For a deeper dive into configurable options, skim the highlights below and see [`OpenProvenceModel.process`](https://github.com/hotchpotch/open_provence/blob/main/open_provence/modeling_open_provence_standalone.py) for the complete API. PyTorch may suggest enabling `torch.set_float32_matmul_precision("high")` to leverage TF32 tensor cores; inference still succeeds with the default setting.

### 🔧 Key `process()` arguments

`process()` handles single queries, batched queries, and nested document structures. The most commonly tuned arguments are:

- **`question: str | Sequence[str]`** – Query text. Provide a list to batch multiple questions; each item pairs with the corresponding entry in `context`.
- **`context: str | Sequence[str] | Sequence[Sequence[str]]`** – Contexts aligned to the query. Use a list for one document per query, or a list of lists to supply multiple documents (or pre-split sentences) for each query.
- **`title: str | Sequence[str] | Sequence[Sequence[str]] | None`** – Optional titles. The default sentinel `"first_sentence"` marks the opening sentence so it can be forced to stay when combined with `always_select_title=True` or `first_line_as_title=True`; without those flags it behaves like any other sentence. Set `None` to disable all title handling.
- **`threshold: float` (default `0.1`)** – Pruning probability threshold. Larger values discard more sentences; values in `0.05–0.5` work well across datasets.
- **`batch_size: int` (default `32`)** – Number of contexts processed per inference batch. Increase for higher throughput, decrease if you run out of memory.
- **`language: str | None`** – Choose the built-in splitter (`"ja"`, `"en"`, or `"auto"`). The default is `None`, which behaves like `"auto"` and detects Japanese vs. English automatically.
- **`reorder: bool` & `top_k: int | None`** – When `reorder=True`, contexts are sorted by reranker score. Combine with `top_k` to keep only the highest-scoring documents.
- **`first_line_as_title: bool` / `always_select_title: bool`** – Extract the first non-empty line as a title and optionally guarantee that the title sentence survives pruning.
- **`return_sentence_metrics: bool` / `return_sentence_texts: bool`** – Include per-sentence probabilities and the lists of kept/removed sentences in the output (useful for analysis tooling).

Additional parameters for debugging, custom splitters, preprocessing workers, and span-level outputs are documented inline in [`OpenProvenceModel.process`](https://github.com/hotchpotch/open_provence/blob/main/open_provence/modeling_open_provence_standalone.py).

> ⚠️ **Common pitfall**: `question` and `context` must have matching shapes. Providing `question: str` with `context: List[str]` is treated as *one* query with multiple documents. To batch independent pairs, use `question: List[str]` and `context: List[str]`. When you pass `context: List[List[str]]`, the inner lists are assumed to be pre-split sentences and the sentence splitter is skipped—use this form only if you have already segmented the text yourself.

## 🧰 Environment Setup

### Base environment (Linux GPU / CUDA 12.8 default)

Run `uv sync`. By default uv now enables the `dev` and `cuda` dependency groups, so the resolver pulls
`torch==2.8.0+cu128` and the matching `nvidia-*` runtime wheels from the `torch-cu128` index whenever
you're on Linux x86_64. Make sure your NVIDIA driver supports CUDA 12.8 (driver ≥ 550.54) before
activating the environment.

- Add FlashAttention during the initial sync with `uv sync --group flash-attn` (the `cuda` group is
  already active).
  (If you need FlashAttention later, re-run `uv sync --group flash-attn` after the base sync.)

### CPU / Metal hosts

If you are on CPU-only Linux, Windows, or macOS, opt out of the CUDA group explicitly:

```bash
uv sync --no-default-groups --group dev --group cpu
```

The same flag combination keeps the resolver on the CPU/Metal `torch==2.8.0` wheel; rerun it whenever
you need to refresh a CPU-only environment.

### Migrating an existing CPU environment to CUDA

If you previously synced the CPU environment and want to flip it to CUDA without recreating the venv,
install the GPU wheel directly:

```bash
uv pip install --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match "torch==2.8.0+cu128"
```

This command also installs the matching `nvidia-*` runtime libraries.

### FlashAttention kernels (optional)

- Using FlashAttention speeds up training and inference.
- Fresh install: `uv sync --group flash-attn`.
- If the PyPI extra works on your GPU but you prefer to keep `uv sync` vanilla, run `uv sync` first and
  then `uv sync --group flash-attn` to add the kernels.
- If you prefer an official wheel: download the match for your platform from https://github.com/Dao-AILab/flash-attention/releases (e.g. save it under `./tmp/`) and install with `uv pip install ./tmp/<wheel-name.whl>`.
- If you maintain a vetted wheel locally: `uv pip install ./tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`.


## 📊 Evaluation Summary

The full breakdown lives in the [OpenProvence v1 Evaluation Report](docs/eval_reports/open_provence_v1_eval_report.md). Key takeaways:

### MLDR (English, LLM evaluation on the MLDR dataset)

- Baseline (no pruning) records Has Answer 93.68%.
- `xsmall` @ th=0.05 maintains 93.68% Has Answer with 82.18% positive / 99.18% negative compression, while remaining the fastest Provence checkpoint.
- `base` @ th=0.05 also keeps Has Answer at 93.68% and deepens compression to 90.05% positive / 99.62% negative.
- `large` @ th=0.10 reaches 93.10% Has Answer with 94.38% positive / 99.90% negative compression, matching the naver/provence baseline’s retention while remaining fully open-source, fine-tunable, and comparable in size (310M vs. 305M parameters).
- `naver-provence` @ th=0.05 (reference) posts 93.10% Has Answer with 92.10% positive / 99.15% negative compression.

### MLDR (Japanese, LLM evaluation on the MLDR dataset)

- Baseline (no pruning) records Has Answer 77.71%.
- `xsmall` @ th=0.05 lifts Has Answer to 81.93% with 76.46% positive / 96.11% negative compression.
- `base` @ th=0.05 delivers the strongest result: 83.13% Has Answer and 80.98% positive / 97.89% negative compression.
- `large` @ th=0.10 balances 79.52% Has Answer with 87.89% positive / 98.82% negative compression.

### Dataset Benchmarks (Mean Across QA Suites, th=0.10)

- English configuration: `en-gte` (F2 0.734, 39.9% compression, 0.55 s), `xsmall` (F2 0.696, 33.8%, 0.34 s), `base` (F2 0.737, 39.9%, 0.69 s), `large` (F2 0.749, 41.7%, 1.04 s).
- Japanese configuration: `xsmall` (F2 0.727, 53.2%, 0.32 s), `base` (F2 0.768, 57.4%, 1.06 s), `large` (F2 0.783, 59.1%, 1.69 s).

## 🎓 Model Training

### Quick Start: Minimal Training

```bash
# English model training example
uv run open_provence_trainer configs/toy-open-provence-reranker-v1-gte-modernbert-base.yaml

# Japanese model training example
uv run open_provence_trainer configs/toy-open-provence-reranker-v1.yaml
```

These toy configurations reach usable pruning quality despite the tiny datasets. On an RTX 5090 they finish in roughly 5–10 minutes (including nano evaluations). While they do not match the full OpenProvence v1 checkpoints, they are perfect for smoke-testing the training pipeline end to end.

### Full Training

For detailed training instructions, see [docs/train.md](docs/train.md).

### Software Testing, Formatting, Type Checking, etc.

```bash
uv run tox
```

## 📊 Dataset Creation

We provide end-to-end scripts for building Provence-style datasets from your own domain data. Adapting the pipeline to business- or research-specific corpora is straightforward—follow the instructions in [docs/create_context_relevance_dataset.md](docs/create_context_relevance_dataset.md).

## 📈 Evaluation

### Cross-Dataset Evaluation

Use this script suite to measure retention across multiple QA datasets; see [docs/eval_dataset.md](docs/eval_dataset.md) for configuration details.

### MLDR Benchmark Evaluation

Evaluation on long-document retrieval benchmarks. For details, see [docs/eval_mldr.md](docs/eval_mldr.md).

## 📄 License

- MIT License

Model weights, training and inference code, plus dataset creation tooling are published under the MIT License. Refer to each dataset card for its specific licensing terms.

## 🙏 Acknowledgments

We deeply appreciate the following research and projects in developing this project:

### Provence Paper & Implementation

[Provence: efficient and robust context pruning for retrieval-augmented generation](https://arxiv.org/abs/2501.16214)

We are grateful to the Provence authors at [Naver Labs Europe](https://europe.naverlabs.com/) for publishing both the paper and the accompanying implementation, including the [naver/provence-reranker-debertav3-v1](https://huggingface.co/naver/provence-reranker-debertav3-v1) checkpoint. Their public release makes it possible to verify just how strong Provence-style pruning can be in practice, and it directly inspired this project.

### Sentence Transformers

[Sentence Transformers](https://github.com/huggingface/sentence-transformers)

This project's training scripts were created with reference to the Sentence Transformers CrossEncoder implementation. We appreciate the developers who publish useful code to the open-source community and maintain it continuously.

## 📝 Citation

```bibtex
@misc{yuichi-tateno-2025-open-provence,
  url = {https://github.com/hotchpotch/open_provence},
  title = {OpenProvence: An Open-Source Implementation of Efficient and Robust Context Pruning for Retrieval-Augmented Generation},
  author = {Yuichi Tateno},
  year = {2025}
}
```

## 👤 Author

Yuichi Tateno ([@hotchpotch](https://github.com/hotchpotch))
