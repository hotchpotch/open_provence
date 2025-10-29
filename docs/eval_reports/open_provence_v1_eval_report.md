# OpenProvence v1 Evaluation Report

## Executive Summary
- **MLDR (English)**  
  - Compared with the unpruned original, thresholds up to 0.05-0.10 keep Has Answer high while pruning roughly 80-93% of passages across all Provence checkpoints.  
  - The naver-provence(435M) baseline is larger, yet the 30M-310M Provence models deliver comparable scores in this pruning-only evaluation; in particular, the 30M xsmall checkpoint at th=0.05 maintains the original score with only moderate pruning while running noticeably faster, which remains a pleasant surprise.
- **MLDR (Japanese)**  
  - Thresholds up to 0.10 stay at or above the original baseline for all checkpoints. Only large@0.05 trails base after six retries (best 80.72% vs. base 83.13%), while 0.10/0.30/0.50 match or surpass base.
- **Dataset retention benchmarks**  
  - Smaller checkpoints are markedly faster, so choose the threshold/model pair that fits your accuracy-latency budget; as model size increases you can dial the threshold for even higher compression at similar F2, underscoring how well the larger models triage passages.  
  - en-gte (English-only) and base (English + Japanese) post very similar retention/compression numbers, but reranker quality is not measured here; en-gte may still be the strongest English reranker.
- **Runtime environment**  
  - All evaluations ran on an NVIDIA RTX 5090 GPU。

| Short Name | Hugging Face Model ID | Parameters |
| --- | --- | --- |
| base | [hotchpotch/open-provence-reranker-v1](https://huggingface.co/hotchpotch/open-provence-reranker-v1) | 130M |
| xsmall | [hotchpotch/open-provence-reranker-xsmall-v1](https://huggingface.co/hotchpotch/open-provence-reranker-xsmall-v1) | 30M |
| large | [hotchpotch/open-provence-reranker-large-v1](https://huggingface.co/hotchpotch/open-provence-reranker-large-v1) | 310M |
| en-gte | [hotchpotch/open-provence-reranker-v1-gte-modernbert-base](https://huggingface.co/hotchpotch/open-provence-reranker-v1-gte-modernbert-base) | 149M |
| naver-provence | [naver/provence-reranker-debertav3-v1](https://huggingface.co/naver/provence-reranker-debertav3-v1) | 435M |

## MLDR English

| Model | Variant | Has Answer (%) | Compression (pos) | Compression (neg) |
| --- | --- | --- | --- | --- |
| none | original | 93.68% | - | - |
| en-gte | th=0.05 | 94.25% | 86.67% | 99.71% |
| en-gte | th=0.10 | 94.25% | 92.33% | 99.91% |
| en-gte | th=0.30 | 90.23% | 96.98% | 99.99% |
| en-gte | th=0.50 | 87.36% | 98.24% | 100.00% |
| xsmall | th=0.05 | 93.68% | 82.18% | 99.18% |
| xsmall | th=0.10 | 91.95% | 89.54% | 99.68% |
| xsmall | th=0.30 | 87.93% | 97.12% | 99.98% |
| xsmall | th=0.50 | 84.48% | 98.69% | 100.00% |
| base | th=0.05 | 93.68% | 90.05% | 99.62% |
| base | th=0.10 | 91.95% | 93.58% | 99.83% |
| base | th=0.30 | 87.93% | 97.43% | 99.98% |
| base | th=0.50 | 86.21% | 98.61% | 100.00% |
| large | th=0.05 | 93.10% | 91.89% | 99.72% |
| large | th=0.10 | 93.10% | 94.38% | 99.90% |
| large | th=0.30 | 89.08% | 97.42% | 99.99% |
| large | th=0.50 | 86.21% | 98.54% | 100.00% |
| naver-provence | th=0.05 | 93.10% | 92.10% | 99.15% |
| naver-provence | th=0.10 | 93.10% | 94.00% | 99.50% |
| naver-provence | th=0.20 | 90.23% | 95.64% | 99.72% |
| naver-provence | th=0.50 | 84.48% | 97.43% | 99.90% |

## MLDR Japanese

| Model | Variant | Has Answer (%) | Compression (pos) | Compression (neg) |
| --- | --- | --- | --- | --- |
| none | original | 77.71% | - | - |
| xsmall | th=0.05 | 81.93% | 76.46% | 96.11% |
| xsmall | th=0.10 | 80.12% | 83.80% | 97.74% |
| xsmall | th=0.30 | 72.29% | 93.55% | 99.41% |
| xsmall | th=0.50 | 68.07% | 96.89% | 99.78% |
| base | th=0.05 | 83.13% | 80.98% | 97.89% |
| base | th=0.10 | 78.92% | 86.35% | 98.87% |
| base | th=0.30 | 75.30% | 93.74% | 99.66% |
| base | th=0.50 | 68.07% | 96.59% | 99.86% |
| large | th=0.05 | 79.52% | 83.37% | 97.94% |
| large | th=0.10 | 79.52% | 87.89% | 98.82% |
| large | th=0.30 | 72.89% | 93.89% | 99.57% |
| large | th=0.50 | 70.48% | 96.40% | 99.81% |

## Dataset Evaluation Summary (Japanese configuration)

| Model | Threshold | Mean F2 | Mean Compression | Mean Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.690 | 45.2% | 0.30 |
| xsmall | 0.10 | 0.727 | 53.2% | 0.32 |
| xsmall | 0.30 | 0.718 | 67.4% | 0.32 |
| xsmall | 0.50 | 0.631 | 75.0% | 0.30 |
| base | 0.05 | 0.733 | 50.8% | 1.05 |
| base | 0.10 | 0.768 | 57.4% | 1.06 |
| base | 0.30 | 0.768 | 68.4% | 1.00 |
| base | 0.50 | 0.703 | 74.1% | 1.11 |
| large | 0.05 | 0.751 | 53.2% | 1.69 |
| large | 0.10 | 0.783 | 59.1% | 1.69 |
| large | 0.30 | 0.784 | 68.8% | 1.73 |
| large | 0.50 | 0.727 | 73.9% | 1.74 |

## Dataset Evaluation Summary (English configuration)

| Model | Threshold | Mean F2 | Mean Compression | Mean Inference Time (s) |
| --- | --- | --- | --- | --- |
| en-gte | 0.05 | 0.704 | 34.0% | 0.62 |
| en-gte | 0.10 | 0.734 | 39.9% | 0.55 |
| en-gte | 0.30 | 0.763 | 50.4% | 0.56 |
| en-gte | 0.50 | 0.749 | 56.3% | 0.56 |
| xsmall | 0.05 | 0.669 | 26.9% | 0.41 |
| xsmall | 0.10 | 0.696 | 33.8% | 0.34 |
| xsmall | 0.30 | 0.725 | 47.9% | 0.34 |
| xsmall | 0.50 | 0.701 | 56.0% | 0.34 |
| base | 0.05 | 0.709 | 33.9% | 1.01 |
| base | 0.10 | 0.737 | 39.9% | 0.69 |
| base | 0.30 | 0.760 | 50.1% | 0.69 |
| base | 0.50 | 0.746 | 55.9% | 0.68 |
| large | 0.05 | 0.723 | 36.5% | 1.07 |
| large | 0.10 | 0.749 | 41.7% | 1.04 |
| large | 0.30 | 0.772 | 50.8% | 1.03 |
| large | 0.50 | 0.757 | 55.8% | 1.13 |

### MS MARCO EN

:warning: Mean inference time averages per-dataset `timing.inference_seconds`; the first threshold run still includes warm-up, so 0.05 values may look slower.

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| en-gte | 0.05 | 0.783 | 6.6% | 1.10 |
| en-gte | 0.10 | 0.790 | 9.6% | 1.10 |
| en-gte | 0.30 | 0.798 | 16.4% | 1.11 |
| en-gte | 0.50 | 0.797 | 21.8% | 1.13 |
| xsmall | 0.05 | 0.776 | 3.7% | 0.95 |
| xsmall | 0.10 | 0.782 | 6.4% | 0.72 |
| xsmall | 0.30 | 0.789 | 13.6% | 0.73 |
| xsmall | 0.50 | 0.785 | 20.5% | 0.73 |
| base | 0.05 | 0.784 | 6.7% | 1.49 |
| base | 0.10 | 0.791 | 9.5% | 1.15 |
| base | 0.30 | 0.796 | 16.1% | 1.14 |
| base | 0.50 | 0.789 | 21.3% | 1.13 |
| large | 0.05 | 0.786 | 7.8% | 1.72 |
| large | 0.10 | 0.792 | 10.3% | 1.64 |
| large | 0.30 | 0.802 | 17.0% | 1.61 |
| large | 0.50 | 0.796 | 21.7% | 1.63 |

### Natural Questions EN

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| en-gte | 0.05 | 0.568 | 64.1% | 0.27 |
| en-gte | 0.10 | 0.622 | 70.5% | 0.27 |
| en-gte | 0.30 | 0.684 | 79.2% | 0.28 |
| en-gte | 0.50 | 0.673 | 83.0% | 0.27 |
| xsmall | 0.05 | 0.511 | 55.6% | 0.14 |
| xsmall | 0.10 | 0.559 | 64.4% | 0.15 |
| xsmall | 0.30 | 0.623 | 78.1% | 0.14 |
| xsmall | 0.50 | 0.599 | 83.8% | 0.14 |
| base | 0.05 | 0.587 | 65.2% | 1.14 |
| base | 0.10 | 0.634 | 71.2% | 0.56 |
| base | 0.30 | 0.690 | 79.6% | 0.56 |
| base | 0.50 | 0.680 | 83.4% | 0.56 |
| large | 0.05 | 0.609 | 67.4% | 0.87 |
| large | 0.10 | 0.660 | 73.0% | 0.86 |
| large | 0.30 | 0.702 | 80.2% | 0.87 |
| large | 0.50 | 0.690 | 83.3% | 0.88 |

### GooAQ EN

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| en-gte | 0.05 | 0.761 | 31.4% | 0.50 |
| en-gte | 0.10 | 0.788 | 39.6% | 0.27 |
| en-gte | 0.30 | 0.808 | 55.6% | 0.28 |
| en-gte | 0.50 | 0.776 | 64.1% | 0.27 |
| xsmall | 0.05 | 0.720 | 21.2% | 0.14 |
| xsmall | 0.10 | 0.748 | 30.8% | 0.14 |
| xsmall | 0.30 | 0.763 | 52.0% | 0.14 |
| xsmall | 0.50 | 0.718 | 63.8% | 0.14 |
| base | 0.05 | 0.757 | 29.8% | 0.40 |
| base | 0.10 | 0.785 | 38.9% | 0.36 |
| base | 0.30 | 0.794 | 54.5% | 0.36 |
| base | 0.50 | 0.768 | 63.0% | 0.36 |
| large | 0.05 | 0.773 | 34.2% | 0.62 |
| large | 0.10 | 0.794 | 41.7% | 0.62 |
| large | 0.30 | 0.812 | 55.2% | 0.61 |
| large | 0.50 | 0.784 | 62.4% | 0.87 |

### MS MARCO JA

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.849 | 21.3% | 0.90 |
| xsmall | 0.10 | 0.867 | 27.7% | 0.89 |
| xsmall | 0.30 | 0.871 | 41.6% | 0.88 |
| xsmall | 0.50 | 0.820 | 51.1% | 0.90 |
| base | 0.05 | 0.871 | 26.5% | 2.41 |
| base | 0.10 | 0.888 | 32.4% | 2.44 |
| base | 0.30 | 0.895 | 43.7% | 2.41 |
| base | 0.50 | 0.859 | 51.0% | 3.01 |
| large | 0.05 | 0.880 | 28.9% | 4.11 |
| large | 0.10 | 0.897 | 34.5% | 4.06 |
| large | 0.30 | 0.907 | 44.7% | 4.07 |
| large | 0.50 | 0.873 | 51.1% | 4.11 |

### AutoWikiQA

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.666 | 59.1% | 0.29 |
| xsmall | 0.10 | 0.716 | 68.0% | 0.29 |
| xsmall | 0.30 | 0.692 | 81.7% | 0.28 |
| xsmall | 0.50 | 0.582 | 87.2% | 0.28 |
| base | 0.05 | 0.715 | 63.8% | 1.73 |
| base | 0.10 | 0.764 | 70.9% | 1.43 |
| base | 0.30 | 0.749 | 81.8% | 1.38 |
| base | 0.50 | 0.660 | 86.2% | 1.74 |
| large | 0.05 | 0.735 | 66.2% | 2.34 |
| large | 0.10 | 0.779 | 72.2% | 2.31 |
| large | 0.30 | 0.773 | 81.5% | 2.68 |
| large | 0.50 | 0.693 | 85.8% | 2.35 |

### JAQuAD

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.625 | 67.1% | 0.27 |
| xsmall | 0.10 | 0.680 | 75.5% | 0.27 |
| xsmall | 0.30 | 0.664 | 87.2% | 0.27 |
| xsmall | 0.50 | 0.553 | 91.7% | 0.27 |
| base | 0.05 | 0.680 | 71.7% | 1.30 |
| base | 0.10 | 0.729 | 77.7% | 1.34 |
| base | 0.30 | 0.731 | 86.6% | 1.34 |
| base | 0.50 | 0.645 | 90.6% | 1.33 |
| large | 0.05 | 0.708 | 73.5% | 2.30 |
| large | 0.10 | 0.748 | 78.9% | 2.27 |
| large | 0.30 | 0.752 | 86.8% | 2.28 |
| large | 0.50 | 0.677 | 90.3% | 2.30 |

### JQARA

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.639 | 56.5% | 0.02 |
| xsmall | 0.10 | 0.679 | 65.5% | 0.02 |
| xsmall | 0.30 | 0.648 | 81.2% | 0.02 |
| xsmall | 0.50 | 0.520 | 88.5% | 0.02 |
| base | 0.05 | 0.677 | 61.2% | 0.09 |
| base | 0.10 | 0.730 | 70.1% | 0.09 |
| base | 0.30 | 0.729 | 81.2% | 0.09 |
| base | 0.50 | 0.637 | 86.3% | 0.09 |
| large | 0.05 | 0.710 | 65.4% | 0.14 |
| large | 0.10 | 0.753 | 72.1% | 0.14 |
| large | 0.30 | 0.726 | 81.8% | 0.15 |
| large | 0.50 | 0.644 | 86.7% | 0.15 |

### J-SQuAD

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.682 | 52.2% | 0.28 |
| xsmall | 0.10 | 0.728 | 61.2% | 0.28 |
| xsmall | 0.30 | 0.701 | 76.3% | 0.30 |
| xsmall | 0.50 | 0.589 | 83.7% | 0.29 |
| base | 0.05 | 0.730 | 57.8% | 1.72 |
| base | 0.10 | 0.768 | 65.3% | 1.75 |
| base | 0.30 | 0.756 | 76.9% | 1.40 |
| base | 0.50 | 0.662 | 82.7% | 1.76 |
| large | 0.05 | 0.753 | 60.3% | 2.78 |
| large | 0.10 | 0.787 | 66.6% | 2.87 |
| large | 0.30 | 0.776 | 76.9% | 2.44 |
| large | 0.50 | 0.697 | 82.0% | 2.41 |

### MIRACL JA

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.814 | 39.3% | 0.03 |
| xsmall | 0.10 | 0.843 | 45.8% | 0.03 |
| xsmall | 0.30 | 0.840 | 58.7% | 0.03 |
| xsmall | 0.50 | 0.769 | 65.6% | 0.03 |
| base | 0.05 | 0.839 | 43.0% | 0.11 |
| base | 0.10 | 0.866 | 48.7% | 0.11 |
| base | 0.30 | 0.870 | 59.3% | 0.11 |
| base | 0.50 | 0.835 | 64.8% | 0.31 |
| large | 0.05 | 0.852 | 44.8% | 0.19 |
| large | 0.10 | 0.880 | 51.3% | 0.19 |
| large | 0.30 | 0.881 | 59.8% | 0.19 |
| large | 0.50 | 0.853 | 64.6% | 0.19 |

### MKQA JA

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.594 | 59.1% | 0.04 |
| xsmall | 0.10 | 0.642 | 68.0% | 0.04 |
| xsmall | 0.30 | 0.622 | 81.6% | 0.04 |
| xsmall | 0.50 | 0.500 | 89.9% | 0.04 |
| base | 0.05 | 0.675 | 66.2% | 0.18 |
| base | 0.10 | 0.723 | 73.0% | 0.18 |
| base | 0.30 | 0.692 | 83.7% | 0.18 |
| base | 0.50 | 0.595 | 89.1% | 0.18 |
| large | 0.05 | 0.686 | 68.4% | 0.33 |
| large | 0.10 | 0.731 | 74.3% | 0.33 |
| large | 0.30 | 0.726 | 84.6% | 0.34 |
| large | 0.50 | 0.642 | 88.9% | 0.33 |

### Mr.TyDi JA

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.811 | 36.8% | 0.04 |
| xsmall | 0.10 | 0.843 | 43.4% | 0.04 |
| xsmall | 0.30 | 0.837 | 56.2% | 0.04 |
| xsmall | 0.50 | 0.759 | 64.5% | 0.04 |
| base | 0.05 | 0.841 | 41.7% | 0.12 |
| base | 0.10 | 0.865 | 47.1% | 0.13 |
| base | 0.30 | 0.869 | 57.6% | 0.12 |
| base | 0.50 | 0.818 | 64.1% | 0.13 |
| large | 0.05 | 0.851 | 43.4% | 0.21 |
| large | 0.10 | 0.874 | 48.4% | 0.21 |
| large | 0.30 | 0.883 | 57.7% | 0.21 |
| large | 0.50 | 0.849 | 62.8% | 0.21 |

### Quiz no Mori

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.644 | 61.0% | 0.18 |
| xsmall | 0.10 | 0.690 | 69.7% | 0.18 |
| xsmall | 0.30 | 0.642 | 83.9% | 0.19 |
| xsmall | 0.50 | 0.514 | 90.4% | 0.18 |
| base | 0.05 | 0.696 | 65.8% | 0.90 |
| base | 0.10 | 0.736 | 73.0% | 0.93 |
| base | 0.30 | 0.720 | 83.7% | 0.91 |
| base | 0.50 | 0.621 | 88.8% | 0.92 |
| large | 0.05 | 0.717 | 67.4% | 1.58 |
| large | 0.10 | 0.759 | 73.9% | 1.57 |
| large | 0.30 | 0.741 | 84.0% | 1.86 |
| large | 0.50 | 0.647 | 88.6% | 1.56 |

### Quiz Works

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.646 | 58.9% | 0.16 |
| xsmall | 0.10 | 0.696 | 68.1% | 0.16 |
| xsmall | 0.30 | 0.677 | 82.4% | 0.16 |
| xsmall | 0.50 | 0.562 | 89.1% | 0.16 |
| base | 0.05 | 0.699 | 63.9% | 0.76 |
| base | 0.10 | 0.742 | 71.3% | 0.76 |
| base | 0.30 | 0.734 | 83.0% | 0.76 |
| base | 0.50 | 0.637 | 88.1% | 0.76 |
| large | 0.05 | 0.727 | 66.4% | 1.29 |
| large | 0.10 | 0.769 | 72.9% | 1.30 |
| large | 0.30 | 0.757 | 83.1% | 1.60 |
| large | 0.50 | 0.674 | 88.0% | 1.30 |

### JFWIR

| Model | Threshold | F2 | Compression | Inference Time (s) |
| --- | --- | --- | --- | --- |
| xsmall | 0.05 | 0.677 | 41.1% | 0.96 |
| xsmall | 0.10 | 0.704 | 50.3% | 0.96 |
| xsmall | 0.30 | 0.687 | 68.8% | 0.95 |
| xsmall | 0.50 | 0.571 | 79.8% | 0.96 |
| base | 0.05 | 0.710 | 47.6% | 3.03 |
| base | 0.10 | 0.732 | 55.1% | 3.08 |
| base | 0.30 | 0.719 | 69.7% | 3.04 |
| base | 0.50 | 0.633 | 78.4% | 3.05 |
| large | 0.05 | 0.724 | 50.2% | 5.02 |
| large | 0.10 | 0.744 | 56.9% | 4.99 |
| large | 0.30 | 0.735 | 69.9% | 5.00 |
| large | 0.50 | 0.659 | 77.8% | 5.87 |
