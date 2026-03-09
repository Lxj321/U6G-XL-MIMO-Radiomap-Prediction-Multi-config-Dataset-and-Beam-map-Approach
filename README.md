# U6G XL-MIMO Radiomap Prediction: Multi-config Dataset & Beam Map Approach

A public benchmark and reproducibility package for **multi-configuration radiomap prediction** in **U6G / XL-MIMO** systems.

> **Public release:** The dataset, pretrained models, source code, and project website are now publicly available.
>
> - **Paper (arXiv):** https://arxiv.org/abs/2603.06401
> - **Dataset & Pretrained Models (Hugging Face):** https://huggingface.co/datasets/lxj321/Multi-config-Radiomap-Dataset
> - **Code Repository (GitHub):** https://github.com/Lxj321/MulticonfigRadiomapDataset
> - **Project Website:** https://lxj321.github.io/MulticonfigRadiomapDataset/

---

## Overview

This project provides a unified benchmark for radiomap prediction under multiple transmitter configurations in U6G / XL-MIMO systems. It is designed to support reproducible research on:

- **multi-configuration radiomap prediction**
- **cross-configuration generalization**
- **cross-environment generalization**
- **beam-aware radiomap modeling**
- **sparse radiomap reconstruction**

A key feature of the release is the joint organization of:

- **height maps**
- **configuration-aware beam maps**
- **ray-tracing radiomap labels**
- **ray-tracing scenes and related assets**
- **UNet / GAN baseline pipelines**

---

## Released Resources

The public release includes:

### 1. Dataset

The dataset contains resources for large-scale U6G / XL-MIMO radiomap prediction, including:

- height maps
- radiomaps
- beam maps
- configuration files
- ray-tracing scenes and simulation-related assets

### 2. Baseline Code

This repository provides code for:

- UNet training and evaluation
- GAN training and evaluation
- dataset preprocessing and loading
- benchmark task execution
- dataset generation pipeline
- beam map generation
- radiomap generation and validation

### 3. Pretrained Models

Pretrained checkpoints for benchmark tasks are released in the Hugging Face repository.

### 4. Documentation

Detailed documentation is provided on the project website, including:

- dataset structure
- benchmark settings
- quickstart instructions
- pretrained model organization

---

## Quick Facts

- **Scenes:** 800
- **Frequencies:** 1.8 / 2.6 / 3.5 / 4.9 / 6.7 GHz
- **TX antennas:** up to 1024 TR
- **Beam counts:** 1 / 8 / 16 / 64
- **Beam pattern:** 3GPP TR 38.901

---

## Benchmark Tasks

The benchmark uses a unified task naming scheme:

- `random_dense_feature`
- `random_dense_encoding`
- `beam_dense_feature`
- `beam_dense_encoding`
- `scene_dense_feature`
- `scene_dense_encoding`
- `random_sparse_feature_samples819`
- `random_sparse_encoding_samples819`

These tasks cover variations in:

- split strategy: random / beam / scene
- supervision density: dense / sparse
- input mode: feature / encoding

---

## Repository Structure

This repository includes code and documentation for:

- dataset generation
- model training
- model evaluation
- project documentation

Main scripts include:

- `DatasetGeneration_Step1_OSMDownload.py`
- `DatasetGeneration_Step2_OSMToSionna.py`
- `DatasetGeneration_Step3_OSMToHeightMap.py`
- `DatasetGeneration_Step4_RadiomapRT.py`
- `DatasetGeneration_Step5_RadiomapValidation.py`
- `DatasetGeneration_Step6_BeammapGenerator.py`
- `ModelTraining_GAN.py`
- `ModelTraining_Unet.py`
- `ModelEvaluation_GAN.py`
- `ModelEvaluation_Unet.py`
- `multiconfig_dataset_preprocess_GAN.py`
- `multiconfig_dataset_preprocess_Unet.py`
- `modules_Unet.py`

---

## Getting Started

### Option A: Inspect the released dataset
Download the dataset packages and pretrained models from Hugging Face:

- https://huggingface.co/datasets/lxj321/Multi-config-Radiomap-Dataset

### Option B: Evaluate released checkpoints
Use the released pretrained models together with the evaluation scripts in this repository.

### Option C: Retrain benchmark models
Use the training scripts together with the documented benchmark task settings.

### Option D: Reproduce data generation
Use the dataset-generation scripts to inspect or reproduce the pipeline for scene processing, radiomap generation, validation, and beam map construction.

For detailed usage instructions, benchmark definitions, and documentation, please refer to:

- **Project Website:** https://lxj321.github.io/MulticonfigRadiomapDataset/

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{li2026u6gxlmimoradiomapprediction,
      title={U6G XL-MIMO Radiomap Prediction: Multi-Config Dataset and Beam Map Approach}, 
      author={Xiaojie Li and Yu Han and Zhizheng Lu and Shi Jin and Chao-Kai Wen},
      year={2026},
      eprint={2603.06401},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2603.06401}
}
```

---

## License

* **Code:** MIT License
* **Dataset:** CC-BY-4.0
* **Pretrained Models:** see the Hugging Face repository for the corresponding release terms

---

## Contact

**Xiaojie Li (李宵杰)**
Email: `xiaojieli@seu.edu.cn` / `xiaojieli@nuaa.edu.cn`

