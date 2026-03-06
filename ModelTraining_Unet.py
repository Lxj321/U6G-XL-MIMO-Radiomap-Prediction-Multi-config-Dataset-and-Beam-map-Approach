"""
Jupyter Notebook 友好的灵活训练脚本 (最终完美版)
- 使用正确的层名匹配：'firstU' 和 'secondU'
- 修复了所有冻结和优化器问题
- 适配Jupyter Notebook
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# CUDA配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.default_generator = torch.Generator(device="cpu")

# ==================== 导入 ====================
from multiconfig_dataset_prepcocess_Unet import MultiBeamRadioDataset
from modules_Unet import RadioWNet
from typing import Literal
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import copy
from collections import defaultdict

# ==================== 🎯 配置区域 ====================
class TrainConfig:
    """训练配置类"""

    # ====== experiment axes ======
    SPLIT_STRATEGY: Literal["random", "beam", "scene"] = "random"
    TASK_MODE: Literal["dense", "sparse"] = "dense"
    USE_FEATURE_MAPS: bool = True  # True: featuremap, False: continuous encoding

    # ========== 训练模式配置 ==========
    TRAIN_MODE = "both"  # 🔧 "both" / "first_only" / "second_only"
    FIRST_UNET_PATH = None  # 🔧 second_only模式需要
    
    # ========== 数据路径配置 ==========
    DIR_MULTIBEAM = "Dataset/radiomaps"
    DIR_HEIGHT_MAPS = "Dataset/height_maps"
    DIR_FEATURE_MAPS = "Dataset/beam_maps"
    
    # ========== 训练参数配置 ==========
    RANDOM_SEED = 42
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    BATCH_SIZE = 256
    
    NUM_EPOCHS_FIRST = 5
    NUM_EPOCHS_SECOND = 5
    
    LEARNING_RATE_FIRST = 1e-4
    LEARNING_RATE_SECOND = 1e-4
    
    
    # ========== 第二阶段配置 ==========
    FREEZE_FIRST_UNET = True  # 🔧 是否冻结第一个U-Net
    
    # ========== 保存配置 ==========
    SAVE_DIR = "My_Net"
    MODEL_PREFIX = "Solution2"

    if USE_FEATURE_MAPS==True:
        EXP_NAME='featuremap'
    else:
        EXP_NAME='continuous'
    
    def get_first_unet_save_path(self):
        return f'{self.SAVE_DIR}/{self.MODEL_PREFIX}_{self.TASK_MODE}_seed{self.RANDOM_SEED}_{self.EXP_NAME}_First_Net.pt'
    
    def get_second_unet_save_path(self):
        return f'{self.SAVE_DIR}/{self.MODEL_PREFIX}_{self.TASK_MODE}_seed{self.RANDOM_SEED}_{self.EXP_NAME}_Second_Net.pt'
    
    def print_config(self):
        print("=" * 80)
        print("训练配置")
        print("=" * 80)
        print(f"训练模式: {self.TRAIN_MODE}")
        if self.TRAIN_MODE == "second_only":
            print(f"第一阶段模型: {self.FIRST_UNET_PATH or '自动查找'}")
        print(f"第一阶段轮数: {self.NUM_EPOCHS_FIRST}")
        print(f"第二阶段轮数: {self.NUM_EPOCHS_SECOND}")
        print(f"第一阶段学习率: {self.LEARNING_RATE_FIRST:.2e}")
        print(f"第二阶段学习率: {self.LEARNING_RATE_SECOND:.2e}")
        print(f"冻结第一U-Net: {self.FREEZE_FIRST_UNET}")
        print(f"批次大小: {self.BATCH_SIZE}")
        print("=" * 80 + "\n")

# ==================== 损失函数 ====================
def calc_loss_dense(pred, target, Validmask, metrics):
    Validmask = Validmask.unsqueeze(1).float()
    pred_masked = pred * Validmask
    target_masked = target * Validmask
    num_valid = Validmask.sum()
    
    if num_valid == 0:
        loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
    else:
        loss = torch.sum((pred_masked - target_masked) ** 2) / num_valid
    
    metrics['loss'] += loss.item() * target.size(0)
    return loss

def calc_loss_sparse(pred, target, samples, metrics, num_samples):
    criterion = nn.MSELoss()
    loss = criterion(samples * pred, samples * target) * (256**2) / num_samples
    metrics['loss'] += loss.item() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.6f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))

# ==================== ⭐ 正确的冻结函数 ====================
def freeze_first_unet(model):
    """
    冻结第一个U-Net的所有参数
    
    模型结构：
    - firstU.*  → 第一个U-Net，需要冻结
    - secondU.* → 第二个U-Net，保持可训练
    
    Returns:
        frozen_count: 冻结的参数数量
        trainable_count: 可训练的参数数量
    """
    frozen_count = 0
    trainable_count = 0
    
    print(f"\n🔒 冻结第一个U-Net的参数...")
    
    for name, param in model.named_parameters():
        # ⭐ 关键：使用 'firstU' 匹配第一个U-Net的参数
        if 'firstU' in name:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1
    
    print(f"  ✅ 冻结 {frozen_count} 个参数组 (firstU.*)")
    print(f"  ✅ 可训练 {trainable_count} 个参数组 (secondU.*)")
    
    # 显示示例
    print(f"\n冻结的参数（示例）:")
    count = 0
    for name, param in model.named_parameters():
        if 'firstU' in name and count < 3:
            print(f"  🔒 {name}")
            count += 1
    
    print(f"\n可训练的参数（示例）:")
    count = 0
    for name, param in model.named_parameters():
        if 'secondU' in name and count < 3:
            print(f"  ✅ {name}")
            count += 1
    
    return frozen_count, trainable_count


def select_condition(inputs, use_feature_maps):
    idx = [1,2,3] if use_feature_maps else [1,2,3,4,6]
    C = inputs.shape[1]
    if max(idx) >= C:
        raise RuntimeError(f"Channel mismatch: need {idx} but got C={C}. "
                           "Check dataset channel definition.")
    return inputs[:, idx, :, :]


# ==================== 训练函数 ====================
def train_model(model, dataloaders, config, WNetPhase="firstU", num_epochs=3):
    """训练模型"""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ⭐ 冻结第一个U-Net（如果需要）
    if config.FREEZE_FIRST_UNET and WNetPhase == "secondU":
        frozen_count, trainable_count = freeze_first_unet(model)
        
        if trainable_count == 0:
            raise ValueError("❌ 错误: 没有可训练的参数！")
    
    # 创建优化器（在冻结后）
    if WNetPhase == "firstU":
        learning_rate = config.LEARNING_RATE_FIRST
    else:
        learning_rate = config.LEARNING_RATE_SECOND
    
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    if len(trainable_params) == 0:
        raise ValueError("❌ 没有可训练的参数！")
    
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print(f"\n优化器配置:")
    print(f"  学习率: {learning_rate:.2e}")
    print(f"  可训练参数数: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(num_epochs):
        print('\n' + '='*60)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('='*60)

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print(f"Learning rate: {param_group['lr']:.2e}")
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            if config.TASK_MODE == "dense":
                for batch_idx, (inputs, targets, vamask) in enumerate(dataloaders[phase]):
                    cond = select_condition(inputs, config.USE_FEATURE_MAPS).to(device)
                    targets = targets.to(device)
                    vamask = vamask.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        [outputs1, outputs2] = model(cond)
                        
                        if WNetPhase == "firstU":
                            loss = calc_loss_dense(outputs1, targets, vamask, metrics)
                        else:
                            loss = calc_loss_dense(outputs2, targets, vamask, metrics)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    epoch_samples += inputs.size(0)
                    
                    if batch_idx % 10 == 0 and phase == 'train':
                        print(f"  Batch {batch_idx}/{len(dataloaders[phase])}: loss={loss.item():.6f}")
            
            elif config.TASK_MODE == "sparse":
                for batch_idx, (inputs, targets, samples, vamask) in enumerate(dataloaders[phase]):

                    inputs1=samples*targets
                    cond = select_condition(inputs, config.USE_FEATURE_MAPS)   # 3通道 or 5通道 
                    combined = torch.cat([inputs1, cond], dim=1)
                    combined=combined.to(device)
                    targets = targets.to(device)
                    vamask = vamask.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        [outputs1, outputs2] = model(combined)
                        
                        if WNetPhase == "firstU":
                            loss = calc_loss_dense(outputs1, targets, vamask, metrics)
                        else:
                            loss = calc_loss_dense(outputs2, targets, vamask, metrics)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    epoch_samples += inputs.size(0)
                    
                    if batch_idx % 10 == 0 and phase == 'train':
                        print(f"  Batch {batch_idx}/{len(dataloaders[phase])}: loss={loss.item():.6f}")

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and epoch_loss < best_loss:
                print(f"💾 Saving best model (val_loss: {best_loss:.6f} → {epoch_loss:.6f})")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(f'⏱️  Epoch time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        torch.cuda.empty_cache()

    print(f'\n✅ Best val loss: {best_loss:.6f}')
    model.load_state_dict(best_model_wts)
    return model

# ==================== 主训练流程 ====================
def main():
    """主训练函数"""
    config = TrainConfig()
    config.print_config()
    
    # 创建数据集
    print("创建数据集...")
    
    Radio_train = MultiBeamRadioDataset(
        phase="train",
        dir_multibeam=config.DIR_MULTIBEAM,
        dir_height_maps=config.DIR_HEIGHT_MAPS,
        dir_feature_maps=config.DIR_FEATURE_MAPS,
        split_strategy=config.SPLIT_STRATEGY,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        mode=config.TASK_MODE,
        random_seed=config.RANDOM_SEED,
        use_3d_buildings=True,
        use_feature_maps=config.USE_FEATURE_MAPS,
        use_continuous_encoding=not config.USE_FEATURE_MAPS
    )
    
    Radio_val = MultiBeamRadioDataset(
        phase="val",
        dir_multibeam=config.DIR_MULTIBEAM,
        dir_height_maps=config.DIR_HEIGHT_MAPS,
        dir_feature_maps=config.DIR_FEATURE_MAPS,
        split_strategy=config.SPLIT_STRATEGY,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        mode=config.TASK_MODE,
        random_seed=config.RANDOM_SEED,
        use_3d_buildings=True,
        use_feature_maps=config.USE_FEATURE_MAPS,
        use_continuous_encoding=not config.USE_FEATURE_MAPS
    )
    
    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1),
        'val': DataLoader(Radio_val, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)
    }
    
    INPUT_CHANNELS = Radio_train.input_channels
    print(f"输入通道数: {INPUT_CHANNELS}\n")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # 阶段1: 训练第一个U-Net
    if config.TRAIN_MODE in ["both", "first_only"]:
        print("=" * 80)
        print("阶段1: 训练第一个U-Net")
        print("=" * 80)
        
        model_first = RadioWNet(inputs=INPUT_CHANNELS, phase="firstU", use_film=False)
        model_first.to(device)
        
        print(f"\n模型参数统计:")
        total_params = sum(p.numel() for p in model_first.parameters())
        trainable_params = sum(p.numel() for p in model_first.parameters() if p.requires_grad)
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        model_first = train_model(model_first, dataloaders, config,
                                  WNetPhase="firstU", num_epochs=config.NUM_EPOCHS_FIRST)
        
        first_unet_path = config.get_first_unet_save_path()
        torch.save(model_first.state_dict(), first_unet_path)
        print(f"\n✅ 第一个U-Net已保存至: {first_unet_path}")
        
        if config.TRAIN_MODE == "first_only":
            print("\n" + "=" * 80)
            print("✅ 第一阶段训练完成！")
            print("=" * 80)
            return
    
    # 阶段2: 训练第二个U-Net
    if config.TRAIN_MODE in ["both", "second_only"]:
        print("\n" + "=" * 80)
        print("阶段2: 训练第二个U-Net")
        print("=" * 80)
        
        if config.TRAIN_MODE == "second_only":
            if config.FIRST_UNET_PATH is None:
                first_unet_path = config.get_first_unet_save_path()
                if not os.path.exists(first_unet_path):
                    raise ValueError(
                        f"❌ 第一阶段模型不存在: {first_unet_path}\n"
                        f"请在配置中设置 FIRST_UNET_PATH 或先训练第一阶段"
                    )
            else:
                first_unet_path = config.FIRST_UNET_PATH
                if not os.path.exists(first_unet_path):
                    raise FileNotFoundError(f"❌ 第一阶段模型不存在: {first_unet_path}")
        else:
            first_unet_path = config.get_first_unet_save_path()
        
        print(f"\n📂 加载第一阶段模型: {first_unet_path}")
        
        model_second = RadioWNet(inputs=INPUT_CHANNELS, phase="secondU", use_film=False)
        model_second.load_state_dict(torch.load(first_unet_path))
        model_second.to(device)
        
        print(f"\n模型参数统计:")
        total_params = sum(p.numel() for p in model_second.parameters())
        print(f"  总参数: {total_params:,}")
        
        model_second = train_model(model_second, dataloaders, config,
                                   WNetPhase="secondU", num_epochs=config.NUM_EPOCHS_SECOND)
        
        second_unet_path = config.get_second_unet_save_path()
        torch.save(model_second.state_dict(), second_unet_path)
        print(f"\n✅ 第二个U-Net已保存至: {second_unet_path}")
    
    # 完成
    print("\n" + "=" * 80)
    print("🎉 训练完成！")
    print("=" * 80)
    
    if config.TRAIN_MODE == "both":
        print(f"\n保存的模型:")
        print(f"  第一阶段: {config.get_first_unet_save_path()}")
        print(f"  第二阶段: {config.get_second_unet_save_path()}")
    elif config.TRAIN_MODE == "second_only":
        print(f"\n保存的模型:")
        print(f"  第二阶段: {config.get_second_unet_save_path()}")

if __name__ == "__main__":
    main()