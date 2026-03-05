```markdown
# Quickstart

## 1) Prepare dataset and pretrained weights

Download the dataset and pretrained models, then place them as:

```text
Dataset/...
Pretrained_Model/...

## 2) Evaluate pretrained GAN (example)
python ModelEvaluation_GAN.py \
  --dataset_root Dataset \
  --ckpt_dir Pretrained_Model/GAN/random_dense_feature

## 3) Evaluate pretrained UNet (example)
python ModelEvaluation_Unet.py \
  --dataset_root Dataset \
  --ckpt_path Pretrained_Model/Unet/Solution1_dense_seed42_First_Net.pt
