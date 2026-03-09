# Benchmark and Tasks

This project defines a unified benchmark for **multi-configuration radiomap prediction** in **U6G / XL-MIMO** systems.

The benchmark is designed to evaluate model performance under different:

- **split strategies**
- **supervision densities**
- **input modes**

A unified task naming scheme is used consistently across the released benchmark documentation, preprocessing pipelines, evaluation scripts, and pretrained GAN task folders.

---

# Supported Tasks

The released benchmark currently includes the following task IDs:

- `random_dense_feature`
- `random_dense_encoding`
- `beam_dense_feature`
- `beam_dense_encoding`
- `scene_dense_feature`
- `scene_dense_encoding`
- `random_sparse_feature_samples819`
- `random_sparse_encoding_samples819`

These task IDs are used consistently in:

- benchmark documentation
- preprocessing and evaluation pipelines
- pretrained GAN folders such as `Pretrained_Model/GAN/<task_id>/`

---

# Task Naming Rule

Each task ID follows the structure:

```text id="lmow0e"
<split>_<density>_<input_mode>
```

or, for sparse tasks:

```text id="8m0k38"
<split>_<density>_<input_mode>_samples<N>
```

where:

* `<split>` defines the dataset split strategy
* `<density>` defines whether the task is dense or sparse
* `<input_mode>` defines the input representation
* `samples<N>` defines the number of sampled observations in sparse settings

---

# Benchmark Dimensions

## 1. Split strategy

The split strategy defines how the training / validation / test sets are separated.

### `random`

Random split over all available samples.

* standard baseline setting
* evaluates generalization under randomly mixed train/test conditions

### `beam`

Split by beam / configuration dimension.

* evaluates **cross-beam** or **cross-configuration** generalization
* harder than random split because the model must generalize across different beam conditions

### `scene`

Split by scene ID (`u1..u800`).

* evaluates **cross-environment** generalization
* tests whether a model trained on one group of environments can generalize to unseen scenes

---

## 2. Supervision density

The density setting defines whether the task uses full or partial observation support.

### `dense`

Dense setting uses full-grid information.

* all available spatial labels or inputs are used
* serves as the standard full-information benchmark

### `sparse`

Sparse setting uses only partial sampled observations.

* the model must reconstruct or predict the full radiomap from sparse samples
* the released sparse benchmark currently includes:

  * `samples819`

---

## 3. Input mode

The input mode defines how transmitter / beam / configuration information is represented.

### `feature`

Uses a **feature-map-based** input representation.

In the released benchmark, this refers to explicitly constructed configuration-aware feature maps, such as beam-map-related side information.

### `encoding`

Uses a **continuous-encoding-based** input representation.

In the released benchmark, this refers to compact configuration channels constructed from normalized continuous parameters instead of explicit feature maps.

---

# Task Table

| Task ID                             | Split Strategy | Density | Input Mode | Meaning                                             |
| ----------------------------------- | -------------- | ------- | ---------- | --------------------------------------------------- |
| `random_dense_feature`              | random         | dense   | feature    | standard dense baseline                             |
| `random_dense_encoding`             | random         | dense   | encoding   | dense baseline with continuous encoding             |
| `beam_dense_feature`                | beam           | dense   | feature    | cross-beam / cross-configuration generalization     |
| `beam_dense_encoding`               | beam           | dense   | encoding   | cross-beam generalization with continuous encoding  |
| `scene_dense_feature`               | scene          | dense   | feature    | cross-scene / cross-environment generalization      |
| `scene_dense_encoding`              | scene          | dense   | encoding   | cross-scene generalization with continuous encoding |
| `random_sparse_feature_samples819`  | random         | sparse  | feature    | sparse reconstruction with 819 sampled observations |
| `random_sparse_encoding_samples819` | random         | sparse  | encoding   | sparse reconstruction with 819 sampled observations |

---

# Sparse Setting: `samples819`

The suffix `samples819` indicates that each sparse sample uses:

* **819 sampled observations**

This setting is intended to evaluate:

* sparse radiomap reconstruction
* limited-measurement prediction
* robustness under reduced observation availability

In the released preprocessing pipelines, sparse samples are generated only over valid propagation regions, and the sparse observation mask is returned separately before being injected into the final model input.

---

# Relation to Released Baselines and Pretrained Models

The benchmark task naming is directly aligned with the released GAN pretrained-model folder structure:

```text id="7q1rq4"
Pretrained_Model/GAN/<task_id>/
```

Examples:

```text id="tkc5i4"
Pretrained_Model/GAN/random_dense_feature/
Pretrained_Model/GAN/scene_dense_encoding/
Pretrained_Model/GAN/random_sparse_feature_samples819/
```

This alignment allows users to directly map:

* benchmark task
* pretrained checkpoint
* evaluation setting

without additional renaming.

For UNet baselines, checkpoint filenames follow a different naming style, while the underlying benchmark settings remain the same.

---

# Relation to Data Preparation

The exact input tensors used by each baseline are defined in the preprocessing scripts:

* `multiconfig_dataset_prepcocess_GAN.py`
* `multiconfig_dataset_prepcocess_Unet.py`

In particular:

* `feature` tasks use feature-map-based inputs
* `encoding` tasks use continuous-encoding-based inputs
* `sparse` tasks return a sparse sampling mask first, and sparse observations are then constructed in the training / evaluation scripts

---

# Recommended Starting Tasks

For first use, the following tasks are recommended:

* `random_dense_feature` — simplest standard baseline
* `scene_dense_feature` — strongest cross-environment generalization test
* `random_sparse_feature_samples819` — sparse reconstruction benchmark

Together, these three tasks provide a good initial coverage of:

* standard prediction
* generalization
* sparse recovery

---

### `Which task should I use?`

- want a standard baseline → `random_dense_feature`
- want cross-environment evaluation → `scene_dense_feature`
- want sparse reconstruction → `random_sparse_feature_samples819`
