# Pretrained Models

This page documents the pretrained checkpoints provided in `Pretrained_Model/`.

The repository currently provides pretrained models for:

- **GAN-based baselines**
- **UNet-based baselines**

---

# GAN

GAN checkpoints are organized as:

```text
Pretrained_Model/GAN/<task_id>/
  best_G.pth
  best_D.pth
  config.json
  eval_results.json
  epoch_history.json
  batch_history.json
```

## Files

### `best_G.pth`

Best generator checkpoint.

This is the main checkpoint used for:

* inference
* evaluation
* pretrained result reproduction

### `best_D.pth`

Best discriminator checkpoint.

This is mainly provided for:

* adversarial training continuation
* full training-state reproduction

### `config.json`

Stores the task configuration used for training/evaluation.

This may include settings such as:

* split strategy
* input mode
* sparse/dense mode
* number of samples
* training hyperparameters

### `eval_results.json`

Stores evaluation results for the pretrained checkpoint.

### `epoch_history.json`

Stores epoch-level training history.

### `batch_history.json`

Stores batch-level training history.

---

## Available GAN Tasks

The currently available pretrained GAN tasks are:

* `beam_dense_encoding`
* `beam_dense_feature`
* `random_dense_encoding`
* `random_dense_feature`
* `random_sparse_encoding_samples819`
* `random_sparse_feature_samples819`
* `scene_dense_encoding`
* `scene_dense_feature`

These task IDs are aligned with the benchmark naming in `benchmark.md`.

---

## Recommended Usage

For evaluation or inference, users typically only need:

* `best_G.pth`
* `config.json`

The folder naming directly indicates the benchmark setting, e.g.:

```text
Pretrained_Model/GAN/random_dense_feature/
Pretrained_Model/GAN/scene_dense_encoding/
Pretrained_Model/GAN/random_sparse_feature_samples819/
```

This makes it straightforward to map:

* benchmark task
* pretrained checkpoint
* evaluation configuration

---

# UNet

UNet checkpoints are stored under:

```text
Pretrained_Model/Unet/
  *.pt
  *.png
  *.csv
```

## Files

### `*.pt`

Serialized UNet checkpoints.

These files contain pretrained model weights for different task settings.

### `*.png`

Visualization files for qualitative result inspection.

These may include:

* prediction visualizations
* comparison plots
* example outputs

### `*.csv`

Summary tables or recorded evaluation results.

---

## Notes

UNet checkpoints currently use a different naming style from the GAN task folders.

For example, checkpoint names may encode:

* dense vs sparse setting
* featuremap vs environment input
* first-stage vs second-stage network
* random seed

A dedicated mapping table from UNet checkpoint names to benchmark tasks will be provided in a future update.

---

# Current Status

* GAN pretrained folders are already organized with **task-aligned naming**
* UNet checkpoints are available, but their naming will be further standardized

This means:

* **GAN checkpoints** can already be used directly with the benchmark task names
* **UNet checkpoints** may require manual mapping based on filename conventions

---

# Suggested Starting Points

For most users, the following pretrained GAN models are recommended as starting points:

* `random_dense_feature` — standard dense baseline
* `scene_dense_feature` — cross-environment generalization benchmark
* `random_sparse_feature_samples819` — sparse reconstruction benchmark

These provide representative coverage of:

* standard supervised prediction
* generalization to unseen scenes
* sparse observation settings

