# 🔍 RobustLens

**How robust are modern vision models? A rigorous statistical evaluation under distribution shift.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)

## Overview

RobustLens is an end-to-end pipeline + interactive web app that evaluates pretrained image classification models under controlled distribution shifts (blur, noise, JPEG compression, weather effects, etc.) and reports:

- **Accuracy** per corruption type and severity with **bootstrap 95% confidence intervals**
- **Paired significance tests** (McNemar's, paired bootstrap) with **multiple-testing correction** (Holm / Benjamini-Hochberg)
- **Calibration analysis**: ECE, reliability diagrams, temperature scaling
- **Conformal prediction**: coverage guarantees vs prediction set size under shift
- **Ensemble analysis**: does model averaging improve robustness? (soft voting, weighted, stacking)
- **Robustness score** with uncertainty quantification

## Models Evaluated

| Model | Architecture | Parameters | ImageNet Top-1 | Est. Inference Speed (CPU)* |
|-------|-------------|------------|----------------|--------------------------|
| ResNet-50 (`resnet50.a1_in1k`) | CNN | 25.6M | 76.1% | ~25 img/s |
| ConvNeXt-Tiny (`convnext_tiny.in12k_ft_in1k`) | Modern CNN | 28.6M | 82.1% | ~15 img/s |
| ViT-B/16 (`vit_base_patch16_224.augreg_in21k_ft_in1k`) | Transformer | 86.6M | 81.1% | ~8 img/s |
| Swin-Tiny (`swin_tiny_patch4_window7_224.ms_in1k`) | Transformer | 28.3M | 81.3% | ~10 img/s |

> *Speed benchmarks run during Phase 2. Full statistical evaluation results will be populated after Phase 3 on Kaggle GPU.*

## Project Structure

```
robustlens/
├── apps/web/            # Next.js interactive dashboard
├── ml/                  # Python evaluation pipeline (uv project)
│   ├── src/robustlens/  # Core library
│   │   ├── corruptions/ # Image corruption functions
│   │   ├── eval/        # Model evaluation
│   │   ├── stats/       # Statistical analysis
│   │   ├── calibration/ # ECE, temperature scaling
│   │   ├── conformal/   # Conformal prediction
│   │   └── ensembles/   # Ensemble methods
│   ├── notebooks/       # Kaggle GPU notebooks
│   ├── scripts/         # CLI entry points
│   └── tests/           # Unit tests
├── results/             # Precomputed JSON/CSV for web app
├── docs/                # Statistical methodology docs
└── README.md
```

## Quick Start

### Web App (local)

```bash
cd apps/web
pnpm install
pnpm dev
```

### ML Pipeline

```bash
cd ml
uv sync
uv run python scripts/run_eval.py --help
```

### Run Tests

```bash
cd ml
uv run pytest
```

## Methodology

See [docs/stats_primer.md](docs/stats_primer.md) for a plain-language explanation of all statistical methods used, and [docs/interview_qa.md](docs/interview_qa.md) for 30 likely interview questions with answers.

## Tech Stack

- **Frontend**: Next.js 15 (App Router), TypeScript, Tailwind CSS, shadcn/ui, Motion, Recharts
- **Backend/ML**: Python 3.11+, PyTorch, timm, torchvision, scikit-learn, scipy, statsmodels
- **Package Management**: uv (Python), pnpm (Node.js)
- **Compute**: Kaggle GPU (T4) for evaluation
- **Hosting**: Vercel (web app), Hugging Face Spaces (optional Gradio demo)

## Limitations

- Evaluated on CIFAR-10 (or subset), not full ImageNet — results may not generalize
- Only classification models; no detection or segmentation
- Corruptions are synthetic; real-world distribution shifts may differ
- GPU budget limited to free-tier Kaggle

## License

MIT
