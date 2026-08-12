# UNet Architecture Analysis

An experimental framework for comparing **UNet**, **UNet++**, and **UNet 3+** through graph structure, theoretical metrics, and semantic segmentation performance.

The project evaluates the three architecture families under matched parameter budgets and across multiple network depths. It extends the methodology in [`no_free_lunch_architectures-main`](no_free_lunch_architectures-main/) from neural architecture analysis to Pascal VOC semantic segmentation.

## Overview

Each experiment combines three complementary views of an architecture:

- **Graph structure** — effective depth and width, parameterized edges, skip connections, and end-to-end paths;
- **Theoretical behavior** — NNGP and NTK spectra, manifold length, and curvature;
- **Empirical behavior** — training dynamics, validation mIoU, pixel accuracy, convergence, and generalization gap.

Theoretical measurements use a DAG-consistent `UNetDAG` surrogate, while segmentation experiments use the standalone models in [`models/`](models/).

## Quick Start

Requirements: Python 3.11 or later, [`uv`](https://docs.astral.sh/uv/), and Pascal VOC 2012.

```bash
uv sync
uv run python smoke_test_end_to_end.py
```

Place the dataset under `dataset/VOC2012/`, or provide its location with `--dataset_root` or the `VOC2012_ROOT` environment variable.

Run all three architectures with a matched 5M parameter budget:

```bash
uv run python main.py --arch all --dataset VOC2012 --depth 5 --param_budget 5M --seeds 42,123,2024
```

Supported parameter budgets are `2M`, `5M`, `10M`, and `15M`; the corresponding channel widths are selected automatically. Use `--theoretical_only` to skip segmentation training.

See [`run.txt`](run.txt) for the main experiment commands and `--help` for the complete command-line interface.

## Visualization

Generate figures for a saved experiment:

```bash
uv run python visualization.py --experiment depth=5 --output_dir figures
```

Compare an architecture across depths:

```bash
uv run python visualization.py --compare_depths --model UNet3Plus --output_dir figures
```

## Repository Structure

```text
Project_unet/
|-- main.py                  # Experiment and inference entry point
|-- models/                  # Empirical segmentation models
|-- unet_dag.py              # DAG models and architecture definitions
|-- theoretical_metrics.py   # NNGP, NTK, length, and curvature
|-- train.py                 # Training and segmentation metrics
|-- datasets.py              # VOC2012 data pipeline
|-- visualization.py         # Result visualization
|-- experiments/             # Saved metrics and checkpoints
`-- figures/                 # Generated plots
```

## Experiment Outputs

Runs store their configuration and results under `experiments/`. Depending on the selected mode, an experiment may contain:

- DAG structural summaries;
- theoretical and empirical metrics in JSON format;
- trained model checkpoints;
- aggregated statistics for multi-seed experiments;
- error reports for incomplete analysis stages.

## Acknowledgements

The architecture-analysis workflow is based on the accompanying [`no_free_lunch_architectures-main`](no_free_lunch_architectures-main/) reference implementation and adapted here for UNet-family semantic segmentation models.
