# Project UNet

This project compares `UNet`, `UNet++`, and `UNet 3+` under matched parameter budgets, combining DAG structural analysis, theoretical metrics, and empirical segmentation training. The overall workflow is adapted from `no_free_lunch_architectures-main` and extended to semantic segmentation.

The main experiment entry point is [main.py](main.py). A standard run executes the following stages in order:

1. DAG structural analysis
2. Theoretical analysis (`NTK`, `NNGP`, manifold complexity, curvature)
3. Empirical training and validation (`loss`, `mIoU`, `pixel accuracy`, etc.)

If you add `--theoretical_only`, the run skips training and only performs the first two stages.

## 1. Environment Setup

This project uses `uv` for dependency management and requires Python `>= 3.11`.

```powershell
uv sync
```

Run the quick smoke test to verify that the environment and dataset are available:

```powershell
uv run python smoke_test_end_to_end.py
```

## 2. Dataset Setup

The project currently supports two datasets:

- `VOC2012`
- `COCOStuff`

Default dataset locations:

- `VOC2012`: `dataset/VOC2012`
- `COCOStuff`: `dataset/COCO-Stuff`

You can override them with environment variables:

```powershell
$env:VOC2012_ROOT="D:/datasets/VOC2012"
$env:COCOSTUFF_ROOT="D:/datasets/COCO-Stuff"
```

Or pass the dataset path explicitly at runtime:

```powershell
uv run python main.py --dataset VOC2012 --dataset_root D:/datasets/VOC2012 --arch UNet --depth 4
```

### VOC2012 Directory Layout

`VOC2012` should match a structure like this. The loader will try a few compatible variants automatically.

```text
dataset/VOC2012/
|- JPEGImages/
|- SegmentationClass/
\- ImageSets/
   \- Segmentation/
      |- train.txt
      \- val.txt
```

## 3. Common Arguments

Key arguments in `main.py`:

- `--arch`: `UNet` / `UNetPlusPlus` / `UNet3Plus` / `all`
- `--dataset`: `VOC2012` / `COCOStuff`
- `--depth`: encoder depth, commonly `3`, `4`, or `5`
- `--param_budget`: `2M` / `5M` / `10M` / `15M`
- `--seeds`: comma-separated random seeds, for example `42,123,2024`
- `--output_dir`: output directory
- `--theoretical_only`: run structural and theoretical analysis only
- `--dataset_root`: override the default dataset path

The default parameter budget is `5M`. Corresponding `base_channels` values are resolved automatically from `_PARAM_BUDGET_TABLE` in [main.py](main.py).

## 4. Reproduce the Existing Experiments in This Repository

The repository already contains three main experiment result folders:

- `experiments/depth=3`
- `experiments/depth=4`
- `experiments/depth=5`

They all correspond to:

- Dataset: `VOC2012`
- Architectures: `UNet`, `UNetPlusPlus`, `UNet3Plus`
- Depths: `3 / 4 / 5`
- Parameter budget: `5M`
- Random seed: `42`

Use the following commands to reproduce those three result sets directly:

```powershell
uv run python main.py --arch all --dataset VOC2012 --depth 3 --param_budget 5M --seeds 42 --output_dir experiments/depth=3
uv run python main.py --arch all --dataset VOC2012 --depth 4 --param_budget 5M --seeds 42 --output_dir experiments/depth=4
uv run python main.py --arch all --dataset VOC2012 --depth 5 --param_budget 5M --seeds 42 --output_dir experiments/depth=5
```

Under the `5M` budget, the automatically selected `base_channels` are:

| depth | UNet | UNet++ | UNet 3+ |
| --- | ---: | ---: | ---: |
| 3 | 100 | 95 | 93 |
| 4 | 49 | 45 | 50 |
| 5 | 24 | 22 | 28 |

## 5. Experiment Commands

### 5.1 Single-Architecture Runs

To run only one architecture, set `--arch` to the target model name:

```powershell
uv run python main.py --arch UNet --dataset VOC2012 --depth 4 --param_budget 5M --seeds 42 --output_dir experiments/unet_depth4_seed42
uv run python main.py --arch UNetPlusPlus --dataset VOC2012 --depth 4 --param_budget 5M --seeds 42 --output_dir experiments/unetpp_depth4_seed42
uv run python main.py --arch UNet3Plus --dataset VOC2012 --depth 4 --param_budget 5M --seeds 42 --output_dir experiments/unet3p_depth4_seed42
```

### 5.2 Multi-Seed Main Experiments

For more stable statistics, run multiple seeds in one command. The following commands correspond to the main experiments at depths `3`, `4`, and `5`:

```powershell
uv run python main.py --arch all --dataset VOC2012 --depth 3 --param_budget 5M --seeds 42,123,2024 --output_dir experiments/main_depth3_3seeds
uv run python main.py --arch all --dataset VOC2012 --depth 4 --param_budget 5M --seeds 42,123,2024 --output_dir experiments/main_depth4_3seeds
uv run python main.py --arch all --dataset VOC2012 --depth 5 --param_budget 5M --seeds 42,123,2024 --output_dir experiments/main_depth5_3seeds
```

When more than one seed is used, the output directory will also include:

```text
aggregated_metrics.json
```

### 5.3 Theoretical-Only Runs

If you only want structural and theoretical metrics without empirical training:

```powershell
uv run python main.py --arch all --dataset VOC2012 --depth 3 --param_budget 5M --seeds 42 --theoretical_only --output_dir experiments/theory_only_depth3
uv run python main.py --arch all --dataset VOC2012 --depth 4 --param_budget 5M --seeds 42 --theoretical_only --output_dir experiments/theory_only_depth4
uv run python main.py --arch all --dataset VOC2012 --depth 5 --param_budget 5M --seeds 42 --theoretical_only --output_dir experiments/theory_only_depth5
```

You can also switch to other parameter budgets:

```powershell
uv run python main.py --arch all --dataset VOC2012 --depth 5 --param_budget 2M --seeds 42 --theoretical_only --output_dir experiments/theory_only_depth5_2M
uv run python main.py --arch all --dataset VOC2012 --depth 5 --param_budget 10M --seeds 42 --theoretical_only --output_dir experiments/theory_only_depth5_10M
uv run python main.py --arch all --dataset VOC2012 --depth 5 --param_budget 15M --seeds 42 --theoretical_only --output_dir experiments/theory_only_depth5_15M
```

### 5.4 Manual `base_channels` Override

If you do not want to use the parameter-budget lookup table, you can override the channel widths explicitly:

```powershell
uv run python main.py --arch all --dataset VOC2012 --depth 5 --seeds 42,123,2024 --base_channels_unet 24 --base_channels_unetpp 22 --base_channels_unet3p 28 --output_dir experiments/manual_channels_depth5
```

### 5.5 COCO-Stuff Runs

To run the same pipeline on `COCOStuff`, switch the dataset name and path:

```powershell
uv run python main.py --arch all --dataset COCOStuff --dataset_root dataset/COCO-Stuff --depth 5 --param_budget 5M --seeds 42 --output_dir experiments/cocostuff_depth5
```

## 6. Visualization Commands

[visualization.py](visualization.py) generates summary plots from experiment results.

### 6.1 List Available Targets

```powershell
uv run python visualization.py --list_experiments
uv run python visualization.py --list_models
```

### 6.2 Generate Figures for All Experiments

```powershell
uv run python visualization.py
```

### 6.3 Generate Figures for a Specific Experiment Folder

```powershell
uv run python visualization.py --experiment depth=3
uv run python visualization.py --experiment depth=4
uv run python visualization.py --experiment depth=5
```

### 6.4 Compare One Architecture Across Depths

```powershell
uv run python visualization.py --compare_depths
uv run python visualization.py --compare_depths --model UNet
uv run python visualization.py --compare_depths --model UNetPlusPlus
uv run python visualization.py --compare_depths --model UNet3Plus
```

Figures are saved to `figures/` by default.

## 7. Inference and Prediction Visualization

After training, you can load a saved model and generate one prediction visualization. Passing the `.pth` file path is the safest option:

```powershell
uv run python main.py --inference --model_path experiments/depth=3/seed42/UNet/trained_model.pth
uv run python main.py --inference --model_path experiments/depth=4/seed42/UNetPlusPlus/trained_model.pth
uv run python main.py --inference --model_path experiments/depth=5/seed42/UNet3Plus/trained_model.pth
```

If the dataset is stored elsewhere, also pass `--dataset_root`:

```powershell
uv run python main.py --inference --model_path experiments/depth=5/seed42/UNet3Plus/trained_model.pth --dataset_root D:/datasets/VOC2012
```

## 8. Output Directory Layout

A typical full experiment folder looks like this:

```text
experiments/
\- depth=4/
   |- dag_structural_analysis_UNet.json
   |- dag_structural_analysis_UNetPlusPlus.json
   |- dag_structural_analysis_UNet3Plus.json
   \- seed42/
      |- UNet/
      |  |- config.json
      |  |- theoretical_metrics.json
      |  |- empirical_metrics.json
      |  \- trained_model.pth
      |- UNetPlusPlus/
      \- UNet3Plus/
```

Files:

- `dag_structural_analysis_*.json`: graph-structure statistics
- `theoretical_metrics.json`: theoretical metrics
- `empirical_metrics.json`: training and validation metrics
- `trained_model.pth`: saved model weights
- `config.json`: full run configuration

## 9. Code Structure

- [main.py](main.py): main experiment entry point
- [train.py](train.py): empirical training and validation
- [theoretical_metrics.py](theoretical_metrics.py): theoretical analysis implementation
- [visualization.py](visualization.py): result visualization
- [datasets.py](datasets.py): dataset registry and loading
- [unet_dag.py](unet_dag.py): DAG-based architecture encoding

## 10. Recommended First Command

If you want to run one complete experiment that matches the repository's existing results, start here:

```powershell
uv run python main.py --arch all --dataset VOC2012 --depth 4 --param_budget 5M --seeds 42 --output_dir experiments/depth=4
```
