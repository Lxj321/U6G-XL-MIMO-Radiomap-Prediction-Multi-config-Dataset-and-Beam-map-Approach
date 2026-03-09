# Pretrained Models

This page documents the pretrained checkpoints released under `Pretrained_Model/`.

The repository provides pretrained models for:

- **GAN-based baselines**
- **UNet-based baselines**

These checkpoints are intended for:

- evaluation of released benchmark tasks
- reproduction of reported pretrained results
- comparison against new baselines under the same benchmark settings

---

# GAN Checkpoints

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

## File roles

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

Stores the task configuration associated with the training run.

Typical contents may include:

* split strategy
* input mode
* sparse / dense mode
* number of samples
* training hyperparameters

### `eval_results.json`

Stores evaluation results for the pretrained checkpoint.

### `epoch_history.json`

Stores epoch-level training history.

### `batch_history.json`

Stores batch-level training history.

---

## Available GAN tasks

The released pretrained GAN tasks include:

* `beam_dense_encoding`
* `beam_dense_feature`
* `random_dense_encoding`
* `random_dense_feature`
* `random_sparse_encoding_samples819`
* `random_sparse_feature_samples819`
* `scene_dense_encoding`
* `scene_dense_feature`

These task IDs are directly aligned with the benchmark naming defined in the Benchmark page.

---

## Recommended GAN usage

For evaluation or inference, users typically only need:

* `best_G.pth`
* `config.json`

The folder naming directly indicates the benchmark task, for example:

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

# UNet Checkpoints

UNet checkpoints are stored under:

```text
Pretrained_Model/Unet/
  *.pt
  *.png
  *.csv
```

## File roles

### `*.pt`

Serialized UNet checkpoints.

These files contain pretrained model weights for different released benchmark settings.

### `*.png`

Visualization files for qualitative result inspection.

These may include:

* prediction visualizations
* comparison plots
* example outputs

### `*.csv`

Summary tables or recorded evaluation results.

---

## UNet checkpoint usage

Unlike the GAN checkpoints, the released UNet checkpoints are **not** organized by `<task_id>` folder names.

Instead, they are evaluated through the released UNet evaluation pipeline, which uses a predefined checkpoint list and corresponding benchmark settings.

This means:

* GAN checkpoints follow **task-aligned folder naming**
* UNet checkpoints follow a **script-aligned checkpoint naming style**
* both still correspond to the same released benchmark dimensions:

  * random / beam / scene
  * dense / sparse
  * feature / encoding

---

## Released UNet model groups

The released UNet evaluation script covers multiple baseline groups, including:

* random-split dense baselines
* random-split sparse baselines
* beam-split dense baselines
* scene-split dense baselines
* feature-map-based baselines
* continuous-encoding-based baselines

Representative released checkpoint groups include:

* `Solution1_environment`
* `Solution1_featuremap`
* `Solution1_continuous`
* `Solution2_sparse_featuremap`
* `Solution2_sparse_continuous`
* `Solution3_1_beam_featuremap`
* `Solution3_1_beam_continuous`
* `Solution3_2_scene_featuremap`
* `Solution3_2_scene_continuous`

These are evaluated directly by the released UNet evaluation script.

---

## Recommended UNet usage

For most users, the recommended workflow is:

1. place the UNet checkpoints under `Pretrained_Model/Unet/`
2. run the released evaluation script:

   ```bash
   python ModelEvaluation_Unet.py
   ```
3. use the generated summary tables and visualizations for inspection

In practice, users do not need to manually reconstruct the task mapping for every checkpoint as long as they use the released evaluation script.

---

# Summary of Checkpoint Organization

| Baseline | Checkpoint organization                                  | Recommended usage                       |
| -------- | -------------------------------------------------------- | --------------------------------------- |
| GAN      | `Pretrained_Model/GAN/<task_id>/`                        | directly map task → folder → checkpoint |
| UNet     | `Pretrained_Model/Unet/*.pt` with script-defined mapping | use released UNet evaluation script     |

---

# Suggested Starting Points

For first use, the following pretrained GAN models are recommended:

* `random_dense_feature` — standard dense baseline
* `scene_dense_feature` — cross-environment generalization benchmark
* `random_sparse_feature_samples819` — sparse reconstruction benchmark

These provide representative coverage of:

* standard supervised prediction
* generalization to unseen scenes
* sparse observation settings

For UNet baselines, the recommended starting point is to run the released UNet evaluation script directly and inspect the predefined evaluated model groups.



---


## Checkpoint-to-task examples


| Baseline | Example checkpoint / folder | Benchmark meaning |
|---|---|---|
| GAN | `random_dense_feature/best_G.pth` | random split + dense + feature |
| GAN | `scene_dense_encoding/best_G.pth` | scene split + dense + encoding |
| UNet | `Solution2_sparse_featuremap` | sparse + feature |
| UNet | `Solution3_2_scene_continuous` | scene + dense + continuous |

