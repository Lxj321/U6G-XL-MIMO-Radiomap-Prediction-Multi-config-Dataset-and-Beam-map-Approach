# Benchmark & Tasks

This project uses a **unified task naming scheme** aligned with the pretrained GAN folder names.

The benchmark is designed to evaluate model performance under different:

- **split strategies**
- **sampling densities**
- **input modes**

---

# Task List

The currently supported benchmark tasks are:

- `random_dense_feature`
- `random_dense_encoding`
- `beam_dense_feature`
- `beam_dense_encoding`
- `scene_dense_feature`
- `scene_dense_encoding`
- `random_sparse_feature_samples819`
- `random_sparse_encoding_samples819`

These task IDs are used consistently in:

- pretrained GAN folders (`Pretrained_Model/GAN/<task_id>/`)
- evaluation scripts
- benchmark documentation

---

# Task Naming Rule

Each task name follows the structure

```text
<split>_<density>_<input_mode>
````

or, for sparse tasks,

```text
<split>_<density>_<input_mode>_samples<N>
```

where:

* `<split>` specifies the dataset split strategy
* `<density>` specifies whether the supervision is dense or sparse
* `<input_mode>` specifies the input representation
* `samples<N>` specifies the number of sampled observations used in sparse settings

---

# Benchmark Dimensions

## 1. Split Strategy

The split strategy defines **how training / validation / test sets are separated**.

### `random`

Random split over samples.

* Standard baseline setting
* Measures generalization under randomly mixed training/testing conditions

### `beam`

Split by beam / configuration dimension.

* Evaluates **cross-beam** or **cross-configuration** generalization
* Harder than random split because the model must generalize across different beam conditions

### `scene`

Split by scene ID (`u1..u800`).

* Evaluates **cross-environment** generalization
* Tests whether a model trained on one set of environments can generalize to unseen scenes

---

## 2. Density

The density setting defines the supervision/input completeness.

### `dense`

Dense setting uses full-grid information.

* All available spatial labels or inputs are used
* This is the standard full-information benchmark

### `sparse`

Sparse setting uses only partial sampled observations.

* The model must reconstruct or predict the full radiomap from sparse samples
* In the current benchmark, sparse tasks include:

  * `samples819`

---

## 3. Input Mode

The input mode defines how the model represents the input features.

### `feature`

Uses the **feature-map based input**.

This typically refers to using explicitly constructed feature maps (e.g., beam-map related features) as model input.

### `encoding`

Uses the **encoding-based input**.

This typically refers to a more compact or alternative representation rather than the explicit feature map.

> Exact tensor definitions depend on the preprocessing pipeline and corresponding model scripts.

---

# Task Table

| Task ID                             | Split Strategy | Density | Input Mode | Notes                                           |
| ----------------------------------- | -------------- | ------- | ---------- | ----------------------------------------------- |
| `random_dense_feature`              | random         | dense   | feature    | Standard dense baseline                         |
| `random_dense_encoding`             | random         | dense   | encoding   | Dense baseline with encoding input              |
| `beam_dense_feature`                | beam           | dense   | feature    | Cross-beam / cross-configuration generalization |
| `beam_dense_encoding`               | beam           | dense   | encoding   | Cross-beam generalization with encoding input   |
| `scene_dense_feature`               | scene          | dense   | feature    | Cross-scene / cross-environment generalization  |
| `scene_dense_encoding`              | scene          | dense   | encoding   | Cross-scene generalization with encoding input  |
| `random_sparse_feature_samples819`  | random         | sparse  | feature    | Sparse reconstruction with 819 samples          |
| `random_sparse_encoding_samples819` | random         | sparse  | encoding   | Sparse reconstruction with 819 samples          |

---

# Sparse Setting: `samples819`

The suffix `samples819` indicates that the sparse setting uses:

* **819 sampled observations** per example

This setting is intended to evaluate:

* sparse radiomap reconstruction
* limited-measurement prediction
* robustness under reduced observation availability

> The exact sampling pattern (random mask / fixed mask / structured sampling) should be documented in the data preprocessing or evaluation scripts.

---

# Relation to Pretrained Models

The benchmark task names are directly aligned with the pretrained GAN directory structure:

```text
Pretrained_Model/GAN/<task_id>/
```

For example:

```text
Pretrained_Model/GAN/random_dense_feature/
Pretrained_Model/GAN/scene_dense_encoding/
Pretrained_Model/GAN/random_sparse_feature_samples819/
```

This allows users to directly map:

* **benchmark task**
* **pretrained checkpoint**
* **evaluation setting**

without additional renaming.

---

# Notes

* The benchmark naming is currently standardized according to the **GAN pretrained model folders**.
* UNet checkpoints may use different file naming conventions, and a mapping table should be provided separately in the pretrained model documentation.
* Exact preprocessing behavior depends on:

  * `multiconfig_dataset_prepcocess_GAN.py`
  * `multiconfig_dataset_prepcocess_Unet.py`

---

# Recommended Usage

For most users, the following tasks are recommended as starting points:

* `random_dense_feature` — simplest standard baseline
* `scene_dense_feature` — strongest cross-environment generalization test
* `random_sparse_feature_samples819` — sparse reconstruction benchmark

These three tasks provide a good initial coverage of:

* standard prediction
* generalization
* sparse recovery


