# Quickstart

This page provides the fastest way to verify the released benchmark resources. It focuses on **evaluation-first usage**: preparing the dataset and pretrained checkpoints, and then running the provided evaluation scripts for the released UNet and GAN baselines.

If your goal is to:
- **inspect the released data**, start from the Dataset page
- **evaluate released checkpoints**, follow this page
- **retrain models or reproduce the generation pipeline**, refer to the code repository and detailed documentation pages

---

# 1. Prepare dataset and pretrained checkpoints

Place the dataset and pretrained models under the project root as:

```text
<project_root>/
  Dataset/
    radiomaps/
    height_maps/
    beam_maps/
    configs/
    sionna_maps/          # optional
  Pretrained_Model/
    GAN/
      ...
    Unet/
      ...
```

The released evaluation scripts assume these default paths.

## Required folders for evaluation

For evaluating pretrained checkpoints, the required folders are:

* `Dataset/radiomaps`
* `Dataset/height_maps`
* `Dataset/beam_maps`
* `Pretrained_Model/Unet`
* `Pretrained_Model/GAN`

The folder `Dataset/sionna_maps` is optional and is only needed if you want to reproduce the ray-tracing generation pipeline.

---

# 2. Recommended first run

For a first verification, the recommended order is:

1. run the released UNet evaluation script
2. run the released GAN evaluation script

This provides a quick sanity check that:

* dataset paths are correct
* pretrained checkpoints are correctly placed
* the evaluation pipeline runs end-to-end

---

# 3. Evaluate pretrained UNet checkpoints

Script:

```text
ModelEvaluation_Unet.py
```

Run:

```bash
python ModelEvaluation_Unet.py
```

The UNet evaluation script uses a built-in configuration class (`EvalConfig`) and does **not** require command-line arguments.

## Default paths

By default, the script uses:

* `Dataset/radiomaps`
* `Dataset/height_maps`
* `Dataset/beam_maps`
* `Pretrained_Model/Unet`

## Default evaluation settings

* `RANDOM_SEED = 42`
* `TRAIN_RATIO = 0.7`
* `VAL_RATIO = 0.1`
* `TEST_RATIO = 0.2`
* `BATCH_SIZE = 64`

## Outputs

Results are saved to:

```text
evaluation_results/
```

Generated outputs include:

* `evaluation_summary_dB.csv`
* `metrics_comparison_dB.png`
* `{model_name}_visualization.png`

## Notes

* metrics are computed in the **dB domain**
* building regions and invalid regions are excluded
* SSIM is computed only on valid regions

## If you want to change settings

`ModelEvaluation_Unet.py` evaluates a predefined model list from `EvalConfig.MODELS`.

To change:

* evaluated checkpoints
* dataset paths
* split strategy
* batch size
* model list

please edit the corresponding fields in `EvalConfig`.

---

# 4. Evaluate pretrained GAN checkpoints

Script:

```text
ModelEvaluation_GAN.py
```

Run:

```bash
python ModelEvaluation_GAN.py
```

The GAN evaluation script automatically scans the experiment folders under:

```text
Pretrained_Model/GAN/
```

and evaluates discovered experiments containing:

```text
best_G.pth
```

## Default paths

By default, the script uses:

* `Dataset/radiomaps`
* `Dataset/height_maps`
* `Dataset/beam_maps`
* `Pretrained_Model/GAN`

## Outputs

Typical outputs include:

* `Pretrained_Model/GAN/evaluation_summary.json`
* `Pretrained_Model/GAN/evaluation_summary.csv`

Per-experiment files may also include:

* `eval_results.json`
* `eval_visualization.png`

## Supported released task folders

The released GAN evaluator is aligned with task folders such as:

* `beam_dense_encoding`
* `beam_dense_feature`
* `random_dense_encoding`
* `random_dense_feature`
* `random_sparse_encoding_samples819`
* `random_sparse_feature_samples819`
* `scene_dense_encoding`
* `scene_dense_feature`

These are automatically detected from experiment folders containing `best_G.pth`.

## Configuration inference

The script infers task settings from one of the following sources:

* predefined `EXPERIMENT_CONFIGS`
* `config.json`
* experiment folder naming rules

## Optional notebook-style usage

The script can also be used interactively. For example:

```python
results = run_evaluation()
results = run_evaluation(single='random_dense_feature')
results = run_evaluation(visualize=True)
```

---

# 5. Common issues

## File not found

Check that the following folders exist exactly as expected:

* `Dataset/radiomaps`
* `Dataset/height_maps`
* `Dataset/beam_maps`
* `Pretrained_Model/GAN`
* `Pretrained_Model/Unet`

## GPU selection

The UNet evaluation script sets:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

If you need another GPU, modify this value in the script.

## Missing Python packages

The evaluation scripts require packages such as:

* `torch`
* `numpy`
* `matplotlib`
* `pandas`
* `scikit-image`

The UNet evaluation script also imports:

* `seaborn`

---

# 6. What this quickstart covers

This page focuses on **evaluation of released checkpoints**.

It does **not** cover in detail:

* retraining the GAN or UNet baselines
* regenerating the dataset assets
* reproducing the complete ray-tracing pipeline

For those workflows, please refer to the repository scripts and the other documentation pages.



---
**Download order**

- download dataset
- download pretrained models
- run UNet evaluation
- run GAN evaluation


