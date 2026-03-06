"""
RME-GAN 改进版训练脚本 v4
支持功能：
1. ✅ 划分策略: random / beam / scene
2. ✅ 采样模式: dense / sparse
3. ✅ 多稀疏度实验支持
4. ✅ Feature Map 开关（True: 使用u0特征图 / False: 使用编码）
5. ✅ 波类型选择: spherical(球面波) / plane(平面波)
6. ✅ 判别器损失使用 Valid Mask
7. ✅ 不同配置自动保存到不同文件夹

输入通道说明：
【use_feature_maps=True】
- Dense模式: [Tx, 高度, FeatureMap(u0)] → 3通道
- Sparse模式: [采样×GT, Tx, 高度, FeatureMap(u0)] → 4通道

【use_feature_maps=False】（选择维度1,2,3,4,6，去掉建筑物和num_beams）
- Dense模式: [Tx, 高度, freq, TR, beam_id] → 5通道
- Sparse模式: [采样×GT, Tx, 高度, freq, TR, beam_id] → 6通道

注意：L1损失只在valid区域计算，不包括建筑物区域
"""

from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import time
import copy
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# 导入改进版数据集
from multiconfig_dataset_prepcocess_GAN import RMEGANDataset, RMEGANMultiSparsityDataset


# ========================================================================
# 网络结构
# ========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        ]
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.block(x)


class RMEGANGenerator(nn.Module):
    def __init__(self, input_channels=4, base_channels=64, num_res_blocks=3, use_dropout=False):
        super().__init__()
        nc = base_channels
        
        self.enc0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, nc, 7, padding=0),
            nn.BatchNorm2d(nc),
            nn.ReLU(True)
        )
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(nc, nc*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(nc*2),
            nn.ReLU(True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(nc*2, nc*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(nc*4),
            nn.ReLU(True)
        )
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(nc*4, use_dropout))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(nc*4, nc*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(nc*2),
            nn.ReLU(True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(nc*2*2, nc, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(True)
        )
        
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc*2, 1, 7, padding=0),
            nn.Tanh()
        )
    
    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        r = self.res_blocks(e2)
        d2 = self.dec2(r)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        out = self.out(d1)
        return out


class RMEGANDiscriminator(nn.Module):
    def __init__(self, input_channels=4, base_channels=64):
        super().__init__()
        nc = base_channels
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channels + 1, nc, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nc, nc*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(nc*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nc*2, nc*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(nc*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nc*4, nc*8, 4, stride=1, padding=1),
            nn.BatchNorm2d(nc*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(nc*8, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, condition, radiomap):
        x = torch.cat([condition, radiomap], dim=1)
        out = self.main(x)
        return out


# ========================================================================
# 训练器（支持dense和sparse模式）
# ========================================================================

class RMEGANTrainer:
    """RME-GAN 训练器 - 支持 dense/sparse 模式和 feature_map/encoding 切换"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"设备: {self.device}")
        
        # 创建数据集
        print("\n创建数据集...")
        datasets = {}
        for phase in ["train", "val"]:
            datasets[phase] = RMEGANDataset(
                phase=phase,
                dir_multibeam=config.DIR_MULTIBEAM,
                dir_height_maps=config.DIR_HEIGHT_MAPS,
                dir_feature_maps=config.DIR_FEATURE_MAPS,
                split_strategy=config.SPLIT_STRATEGY,
                train_ratio=config.TRAIN_RATIO,
                val_ratio=config.VAL_RATIO,
                test_ratio=config.TEST_RATIO,
                mode=config.MODE,
                fix_samples=config.FIX_SAMPLES,
                num_samples_low=config.NUM_SAMPLES_LOW,
                num_samples_high=config.NUM_SAMPLES_HIGH,
                use_feature_maps=config.USE_FEATURE_MAPS,
                random_seed=config.RANDOM_SEED
            )
        
        self.dataloaders = {
            "train": DataLoader(datasets["train"], batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True),
            "val": DataLoader(datasets["val"], batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
        }
        
        # ⭐ 确定输入通道数
        self.base_input_ch = datasets["train"].input_channels
        print(f"✅ 数据集返回通道数: {self.base_input_ch}")
        print(f"✅ 模式: {config.MODE}")
        print(f"✅ 使用Feature Map: {config.USE_FEATURE_MAPS}")

        # 计算实际模型输入通道数
        # 数据集返回的通道:
        # use_feature_maps=True:  [建筑物, Tx, 高度, FeatureMap] = 4通道
        # use_feature_maps=False: [建筑物, Tx, 高度, freq, TR, num_beams, beam_id] = 7通道
        #
        # 训练时的处理:
        # use_feature_maps=True:  选择[1,2,3] -> [Tx, 高度, FeatureMap] = 3通道
        # use_feature_maps=False: 选择[1,2,3,4,6] -> [Tx, 高度, freq, TR, beam_id] = 5通道
        #
        # sparse模式额外加1个采样通道
        
        if config.USE_FEATURE_MAPS:
            # [Tx, 高度, FeatureMap]
            model_input_ch = 3
        else:
            # [Tx, 高度, freq, TR, beam_id] (去掉num_beams)
            model_input_ch = 5
        
        if config.MODE == "sparse":
            model_input_ch += 1  # 加上采样通道
        
        self.model_input_ch = model_input_ch
        print(f"✅ 模型输入通道数: {model_input_ch}")
        
        # 创建模型
        print("\n创建模型...")
        
        self.netG = RMEGANGenerator(
            input_channels=model_input_ch,
            base_channels=config.BASE_CHANNELS,
            num_res_blocks=config.NUM_RES_BLOCKS
        ).to(self.device)
        
        self.netD = RMEGANDiscriminator(
            input_channels=model_input_ch,
            base_channels=config.BASE_CHANNELS
        ).to(self.device)
        
        # 优化器
        self.optimG = optim.Adam(self.netG.parameters(), lr=config.LR_G, 
                                betas=(config.BETA1, config.BETA2))
        self.optimD = optim.Adam(self.netD.parameters(), lr=config.LR_D, 
                                betas=(config.BETA1, config.BETA2))
        
        # 学习率调度
        self.schedulerG = lr_scheduler.StepLR(self.optimG, step_size=30, gamma=0.5)
        self.schedulerD = lr_scheduler.StepLR(self.optimD, step_size=30, gamma=0.5)
        
        # 损失函数
        self.criterion_GAN = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion_L1 = nn.L1Loss()
        
        # 历史记录
        self.history = {
            'train_G': [], 'train_D': [], 'train_G_adv': [], 'train_G_L1': [],
            'train_D_acc': [], 'val': [], 'best_val': float('inf'),
            'batch_history': []
        }
        
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        
        self._print_config()
    
    def _print_config(self):
        """打印配置信息"""
        config = self.config
        print(f"\n⚙️  训练配置:")
        print(f"  划分策略: {config.SPLIT_STRATEGY}")
        print(f"  采样模式: {config.MODE}")
        if config.MODE == "sparse":
            if config.FIX_SAMPLES > 0:
                print(f"  固定采样点: {config.FIX_SAMPLES}")
            else:
                print(f"  随机采样范围: [{config.NUM_SAMPLES_LOW}, {config.NUM_SAMPLES_HIGH}]")
        print(f"  使用 Feature Map: {config.USE_FEATURE_MAPS}")
        if config.USE_FEATURE_MAPS:
            print(f"    → 输入: [Tx, 高度, FeatureMap(u0)]")
        else:
            print(f"    → 输入: [Tx, 高度, freq, TR, beam_id] (去掉num_beams)")
        print(f"  模型输入通道数: {self.model_input_ch}")
        print(f"  生成器学习率: {config.LR_G:.2e}")
        print(f"  判别器学习率: {config.LR_D:.2e}")
        print(f"  L1损失权重: {config.LAMBDA_L1}")
        print(f"  对抗损失权重: {config.LAMBDA_ADV}")
        print(f"  保存目录: {config.SAVE_DIR}")
        
        print(f"\n生成器参数: {sum(p.numel() for p in self.netG.parameters()):,}")
        print(f"判别器参数: {sum(p.numel() for p in self.netD.parameters()):,}")
    
    def _prepare_inputs(self, batch):
        """
        准备输入数据
        
        数据集返回的通道:
        - use_feature_maps=True:  [建筑物(0), Tx(1), 高度(2), FeatureMap(3)]
        - use_feature_maps=False: [建筑物(0), Tx(1), 高度(2), freq(3), TR(4), num_beams(5), beam_id(6)]
        
        训练时选择的通道:
        - use_feature_maps=True:  [1, 2, 3] = [Tx, 高度, FeatureMap]
        - use_feature_maps=False: [1, 2, 3, 4, 6] = [Tx, 高度, freq, TR, beam_id] (去掉num_beams)
        
        sparse模式: 在最前面加上 [采样×GT] 通道
        
        Returns:
            inputs: 模型输入 [B, C, H, W]
            targets: 目标 [B, 1, H, W]
            valids: 有效掩码 [B, H, W]
            samples_mask: 采样掩码（仅sparse模式）[B, 1, H, W]
        """
        if self.config.MODE == "dense":
            inputs, targets, valids = batch
            samples_mask = None
        else:
            inputs, targets, samples_mask, valids = batch
        
        # 移动到GPU
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        valids = valids.to(self.device)
        
        # 选择需要的通道
        if self.config.USE_FEATURE_MAPS:
            # use_feature_maps=True: 选择通道 [1, 2, 3]
            # [Tx, 高度, FeatureMap]
            condition_input = inputs[:, [1, 2, 3], :, :]
        else:
            # use_feature_maps=False: 选择通道 [1, 2, 3, 4, 6]
            # [Tx, 高度, freq, TR, beam_id] (去掉num_beams即通道5)
            condition_input = inputs[:, [1, 2, 3, 4, 6], :, :]
        
        if self.config.MODE == "sparse":
            samples_mask = samples_mask.to(self.device)
            
            # sparse模式：在最前面加上采样通道
            # 采样通道 = samples_mask * targets
            sampled_input = samples_mask * targets  # [B, 1, H, W]
            inputs = torch.cat([sampled_input, condition_input], dim=1)
        else:
            # dense模式：直接使用条件输入
            inputs = condition_input
        
        return inputs, targets, valids, samples_mask
    
    def _compute_masked_gan_loss(self, pred, label, valid_mask):
        """计算带mask的GAN损失"""
        loss_map = self.criterion_GAN(pred, label)
        
        valid_mask_downsampled = F.adaptive_avg_pool2d(
            valid_mask.unsqueeze(1).float(),
            output_size=(pred.size(2), pred.size(3))
        )
        
        masked_loss_map = loss_map * valid_mask_downsampled
        num_valid = valid_mask_downsampled.sum() + 1e-8
        loss = masked_loss_map.sum() / num_valid
        
        return loss
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.netG.train()
        self.netD.train()
        
        metrics = defaultdict(float)
        batch_losses = []
        n_samples = 0
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 准备输入
            inputs, targets, valids, samples_mask = self._prepare_inputs(batch)
            bs = inputs.size(0)
            
            # 标签平滑
            real_label_value = 1.0 - self.config.LABEL_SMOOTHING
            fake_label_value = 0.0 + self.config.LABEL_SMOOTHING * 0.1
            
            real_label = torch.full((bs, 1, 30, 30), real_label_value).to(self.device)
            fake_label = torch.full((bs, 1, 30, 30), fake_label_value).to(self.device)
            
            # 生成假样本
            fake = self.netG(inputs)
            fake_norm = (fake + 1) / 2
            
            # 判别器输入噪声
            if self.config.D_NOISE > 0:
                targets_noisy = targets + torch.randn_like(targets) * self.config.D_NOISE
                targets_noisy = torch.clamp(targets_noisy, 0, 1)
                fake_norm_noisy = fake_norm + torch.randn_like(fake_norm) * self.config.D_NOISE
                fake_norm_noisy = torch.clamp(fake_norm_noisy, 0, 1)
            else:
                targets_noisy = targets
                fake_norm_noisy = fake_norm
            
            # 训练判别器
            if batch_idx % self.config.D_TRAIN_FREQ == 0:
                self.optimD.zero_grad()
                
                pred_real = self.netD(inputs, targets_noisy)
                loss_D_real = self._compute_masked_gan_loss(pred_real, real_label, valids)
                
                pred_fake = self.netD(inputs, fake_norm_noisy.detach())
                loss_D_fake = self._compute_masked_gan_loss(pred_fake, fake_label, valids)
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.optimD.step()
            else:
                with torch.no_grad():
                    pred_real = self.netD(inputs, targets_noisy)
                    loss_D_real = self._compute_masked_gan_loss(pred_real, real_label, valids)
                    pred_fake = self.netD(inputs, fake_norm_noisy.detach())
                    loss_D_fake = self._compute_masked_gan_loss(pred_fake, fake_label, valids)
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
            
            # 训练生成器
            self.optimG.zero_grad()
            
            pred_fake = self.netD(inputs, fake_norm)
            loss_G_adv = self._compute_masked_gan_loss(pred_fake, real_label, valids)
            
            # ⭐ L1损失 - 只在valid区域计算（不包括建筑物区域）
            # valid_mask 是在数据集中定义的，排除了建筑物和无标签区域
            loss_G_L1 = self.criterion_L1(
                fake_norm * valids.unsqueeze(1),
                targets * valids.unsqueeze(1)
            )
            
            loss_G = (self.config.LAMBDA_ADV * loss_G_adv + 
                     self.config.LAMBDA_L1 * loss_G_L1)
            loss_G.backward()
            
            if self.config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.config.GRAD_CLIP)
            
            self.optimG.step()
            
            # 计算准确率
            with torch.no_grad():
                valid_mask_ds = F.adaptive_avg_pool2d(
                    valids.unsqueeze(1).float(), output_size=(30, 30)
                ) > 0.5
                
                pred_real_prob = torch.sigmoid(pred_real)
                pred_fake_prob = torch.sigmoid(self.netD(inputs, fake_norm.detach()))
                
                acc_real = ((pred_real_prob > 0.5) * valid_mask_ds).float().sum() / \
                          (valid_mask_ds.sum() + 1e-8)
                acc_fake = ((pred_fake_prob < 0.5) * valid_mask_ds).float().sum() / \
                          (valid_mask_ds.sum() + 1e-8)
                d_acc = ((acc_real + acc_fake) / 2).item()
            
            # 记录
            batch_info = {
                'epoch': epoch,
                'batch': batch_idx,
                'loss_G': loss_G.item(),
                'loss_G_adv': loss_G_adv.item(),
                'loss_G_L1': loss_G_L1.item(),
                'loss_D': loss_D.item(),
                'D_acc': d_acc
            }
            batch_losses.append(batch_info)
            
            metrics['G'] += loss_G.item() * bs
            metrics['D'] += loss_D.item() * bs
            metrics['G_adv'] += loss_G_adv.item() * bs
            metrics['G_L1'] += loss_G_L1.item() * bs
            metrics['D_acc'] += d_acc * bs
            n_samples += bs
            
            pbar.set_postfix({
                'G': f"{loss_G.item():.4f}",
                'D': f"{loss_D.item():.4f}",
                'L1': f"{loss_G_L1.item():.4f}",
                'Dacc': f"{d_acc:.3f}"
            })
        
        avg_metrics = {k: v/n_samples for k, v in metrics.items()}
        avg_metrics['batch_losses'] = batch_losses
        return avg_metrics
    
    def validate(self):
        """验证"""
        self.netG.eval()
        val_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch in self.dataloaders['val']:
                inputs, targets, valids, _ = self._prepare_inputs(batch)
                
                outputs = self.netG(inputs)
                outputs_norm = (outputs + 1) / 2
                
                # L1损失 - 只在valid区域计算
                loss = self.criterion_L1(
                    outputs_norm * valids.unsqueeze(1),
                    targets * valids.unsqueeze(1)
                )
                
                val_loss += loss.item() * inputs.size(0)
                n_samples += inputs.size(0)
        
        return val_loss / n_samples
    
    def visualize_results(self, epoch, num_samples=3):
        """可视化"""
        self.netG.eval()
        
        with torch.no_grad():
            dataiter = iter(self.dataloaders['val'])
            batch = next(dataiter)
            
            # 准备数据 - 根据模式分别处理
            if self.config.MODE == "dense":
                inputs_full, targets, valids = batch
                samples_mask = None
                
                inputs_full = inputs_full[:num_samples]
                targets = targets[:num_samples].to(self.device)
                valids = valids[:num_samples]
                
                inputs, _, _, _ = self._prepare_inputs(
                    (inputs_full, targets.cpu(), valids)
                )
            else:
                inputs_full, targets, samples_mask, valids = batch
                
                inputs_full = inputs_full[:num_samples]
                targets = targets[:num_samples].to(self.device)
                samples_mask = samples_mask[:num_samples]
                valids = valids[:num_samples]
                
                inputs, _, _, _ = self._prepare_inputs(
                    (inputs_full, targets.cpu(), samples_mask, valids)
                )
            
            outputs = self.netG(inputs)
            outputs_norm = (outputs + 1) / 2
            
            inputs_np = inputs_full.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs_norm.cpu().numpy()
            
            # 确定显示列数
            if self.config.MODE == "sparse" and samples_mask is not None:
                n_cols = 5
                samples_np = samples_mask.cpu().numpy()
            else:
                n_cols = 4
                samples_np = None
            
            fig, axes = plt.subplots(num_samples, n_cols, figsize=(4*n_cols, 4*num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_samples):
                col = 0
                
                # 建筑物
                axes[i, col].imshow(inputs_np[i, 0], cmap='gray')
                axes[i, col].set_title('Building')
                axes[i, col].axis('off')
                col += 1
                
                # Feature Map 或 编码参数
                if self.config.USE_FEATURE_MAPS:
                    if inputs_np.shape[1] > 3:
                        axes[i, col].imshow(inputs_np[i, 3], cmap='viridis')
                        axes[i, col].set_title('FeatureMap')
                        axes[i, col].axis('off')
                        col += 1
                else:
                    if inputs_np.shape[1] > 3:
                        axes[i, col].imshow(inputs_np[i, 3], cmap='viridis')
                        axes[i, col].set_title('Encoding (freq)')
                        axes[i, col].axis('off')
                        col += 1
                
                # 采样点（仅sparse模式）
                if samples_np is not None:
                    axes[i, col].imshow(samples_np[i, 0], cmap='hot')
                    axes[i, col].set_title(f'Samples ({int(samples_np[i, 0].sum())})')
                    axes[i, col].axis('off')
                    col += 1
                
                # Ground Truth
                axes[i, col].imshow(targets_np[i, 0], cmap='viridis', vmin=0, vmax=1)
                axes[i, col].set_title('Ground Truth')
                axes[i, col].axis('off')
                col += 1
                
                # Generated
                axes[i, col].imshow(outputs_np[i, 0], cmap='viridis', vmin=0, vmax=1)
                axes[i, col].set_title('Generated')
                axes[i, col].axis('off')
            
            plt.tight_layout()
            save_path = f"{self.config.SAVE_DIR}/samples_epoch{epoch+1}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✨ 可视化结果已保存到 {save_path}")
    
    def train(self):
        """完整训练"""
        print("\n🚀 开始训练...\n")
        print("=" * 80)
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print("-" * 80)
            
            train_m = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.schedulerG.step()
            self.schedulerD.step()
            
            print(f"\n📊 训练统计:")
            print(f"  生成器 - 总损失: {train_m['G']:.6f}")
            print(f"           对抗损失: {train_m['G_adv']:.6f}")
            print(f"           L1损失: {train_m['G_L1']:.6f}")
            print(f"  判别器 - 损失: {train_m['D']:.6f}")
            print(f"           准确率: {train_m['D_acc']:.4f}", end="")
            
            if train_m['D_acc'] > 0.9:
                print(" ❌ 过高!")
            elif train_m['D_acc'] < 0.3:
                print(" ⚠️  过低!")
            elif 0.5 <= train_m['D_acc'] <= 0.7:
                print(" ✅ 健康")
            else:
                print("")
            
            print(f"  验证损失: {val_loss:.6f}")
            
            # 记录历史
            self.history['train_G'].append(train_m['G'])
            self.history['train_D'].append(train_m['D'])
            self.history['train_G_adv'].append(train_m['G_adv'])
            self.history['train_G_L1'].append(train_m['G_L1'])
            self.history['train_D_acc'].append(train_m['D_acc'])
            self.history['val'].append(val_loss)
            self.history['batch_history'].extend(train_m['batch_losses'])
            
            # 保存最佳模型
            if val_loss < self.history['best_val']:
                self.history['best_val'] = val_loss
                torch.save(self.netG.state_dict(), f"{self.config.SAVE_DIR}/best_G.pth")
                torch.save(self.netD.state_dict(), f"{self.config.SAVE_DIR}/best_D.pth")
                print(f"💾 保存最佳模型 (val={val_loss:.6f})")
            
            # 可视化
            if (epoch + 1) % self.config.VIS_FREQ == 0:
                self.visualize_results(epoch)
            
            # 定期保存
            if (epoch + 1) % self.config.SAVE_FREQ == 0:
                torch.save(self.netG.state_dict(), f"{self.config.SAVE_DIR}/G_ep{epoch+1}.pth")
                self.save_history()
        
        print("\n" + "=" * 80)
        print("✅ 训练完成!")
        self.save_history()
        
        return self.history
    
    def save_history(self):
        """保存训练历史"""
        # 保存batch级别历史
        batch_path = f"{self.config.SAVE_DIR}/batch_history.json"
        with open(batch_path, 'w') as f:
            json.dump(self.history['batch_history'], f, indent=2)
        
        # 保存epoch级别历史
        epoch_history = {k: v for k, v in self.history.items() if k != 'batch_history'}
        epoch_path = f"{self.config.SAVE_DIR}/epoch_history.json"
        with open(epoch_path, 'w') as f:
            json.dump(epoch_history, f, indent=2)
        
        # 保存配置信息
        config_info = {
            'SPLIT_STRATEGY': self.config.SPLIT_STRATEGY,
            'MODE': self.config.MODE,
            'USE_FEATURE_MAPS': self.config.USE_FEATURE_MAPS,
            'FIX_SAMPLES': self.config.FIX_SAMPLES if self.config.MODE == 'sparse' else None,
            'MODEL_INPUT_CH': self.model_input_ch,
            'NUM_EPOCHS': self.config.NUM_EPOCHS,
            'BATCH_SIZE': self.config.BATCH_SIZE,
            'LR_G': self.config.LR_G,
            'LR_D': self.config.LR_D,
            'LAMBDA_L1': self.config.LAMBDA_L1,
            'LAMBDA_ADV': self.config.LAMBDA_ADV
        }
        config_path = f"{self.config.SAVE_DIR}/config.json"
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print(f"📊 历史已保存")


# ========================================================================
# 配置类
# ========================================================================

class Config:
    """训练配置"""
    
    # ========== 数据路径 ==========
    DIR_MULTIBEAM = "Dataset/radiomaps"
    DIR_HEIGHT_MAPS = "Dataset/height_maps"
    DIR_FEATURE_MAPS = "Dataset/beam_maps"
    SAVE_DIR = "rmegan_checkpoints"
    
    # ========== 划分策略 ==========
    SPLIT_STRATEGY = "random"  # "random" / "beam" / "scene"
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    
    # ========== 采样模式 ==========
    MODE = "dense"  # "dense" / "sparse"
    FIX_SAMPLES = 819  # sparse模式下的固定采样点数
    NUM_SAMPLES_LOW = 200
    NUM_SAMPLES_HIGH = 1000
    
    # ========== 模型配置 ==========
    BASE_CHANNELS = 64
    NUM_RES_BLOCKS = 3
    USE_FEATURE_MAPS = True  # True: FeatureMap / False: 编码
    
    # ========== 训练配置 ==========
    BATCH_SIZE = 128
    RANDOM_SEED = 42
    NUM_EPOCHS = 20
    LR_G = 1e-5
    LR_D = 1e-5
    BETA1 = 0.5
    BETA2 = 0.999
    
    # ========== 损失权重 ==========
    LAMBDA_ADV = 0.5
    LAMBDA_L1 = 400.0
    
    # ========== 防护措施 ==========
    LABEL_SMOOTHING = 0.2
    D_NOISE = 0.10
    D_TRAIN_FREQ = 5
    GRAD_CLIP = 5.0
    
    # ========== 保存配置 ==========
    SAVE_FREQ = 5
    VIS_FREQ = 1


# ========================================================================
# 辅助函数：生成保存目录名
# ========================================================================

def generate_save_dir(base_dir, split_strategy, mode, use_feature_maps, fix_samples=None):
    """
    根据配置生成保存目录名
    
    格式: base_dir/{split_strategy}_{mode}_{feature/encoding}[_samples{N}]
    
    示例:
    - random_dense_feature
    - random_sparse_encoding_samples819
    """
    if use_feature_maps:
        feature_str = "feature"
    else:
        feature_str = "encoding"
    
    if mode == "sparse" and fix_samples is not None:
        dir_name = f"{split_strategy}_{mode}_{feature_str}_samples{fix_samples}"
    else:
        dir_name = f"{split_strategy}_{mode}_{feature_str}"
    
    return os.path.join(base_dir, dir_name)


# ========================================================================
# 多稀疏度实验
# ========================================================================

def run_multi_sparsity_experiment(base_config, sparsity_levels=[100, 200, 400, 819, 1000, 2000]):
    """运行多稀疏度实验"""
    print("=" * 80)
    print("🔬 多稀疏度实验")
    print(f"   稀疏度级别: {sparsity_levels}")
    print(f"   使用Feature Map: {base_config.USE_FEATURE_MAPS}")
    print("=" * 80)
    
    results = {}
    
    for sparsity in sparsity_levels:
        print(f"\n{'='*80}")
        print(f"📊 实验: {sparsity} 采样点")
        print(f"{'='*80}")
        
        config = copy.deepcopy(base_config)
        config.MODE = "sparse"
        config.FIX_SAMPLES = sparsity
        config.SAVE_DIR = generate_save_dir(
            base_config.SAVE_DIR,
            config.SPLIT_STRATEGY,
            config.MODE,
            config.USE_FEATURE_MAPS,
            sparsity
        )
        
        trainer = RMEGANTrainer(config)
        history = trainer.train()
        
        results[sparsity] = {
            'best_val': history['best_val'],
            'final_val': history['val'][-1],
            'train_G': history['train_G'][-1],
            'train_D': history['train_D'][-1]
        }
        
        print(f"\n✅ {sparsity} 采样点完成")
        print(f"   最佳验证损失: {history['best_val']:.6f}")
    
    # 保存实验结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_config.USE_FEATURE_MAPS:
        feature_str = "feature"
    else:
        feature_str = "encoding"
    results_path = f"{base_config.SAVE_DIR}/multi_sparsity_{feature_str}_results_{timestamp}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("🎉 多稀疏度实验完成!")
    print("=" * 80)
    
    print("\n📊 实验汇总:")
    print("-" * 50)
    print(f"{'采样点':<10} {'最佳验证损失':<15} {'最终验证损失':<15}")
    print("-" * 50)
    for sparsity, result in results.items():
        print(f"{sparsity:<10} {result['best_val']:<15.6f} {result['final_val']:<15.6f}")
    print("-" * 50)
    
    return results


# ========================================================================
# 主函数
# ========================================================================

def main():
    """主训练函数"""
    print("=" * 80)
    print("🔧 RME-GAN 改进版训练 v4")
    print("✅ 支持 random/beam/scene 划分")
    print("✅ 支持 dense/sparse 模式")
    print("✅ 支持 Feature Map / 编码 切换")
    print("✅ 自动保存到不同目录")
    print("=" * 80)
    
    config = Config()
    
    # 自动生成保存目录
    config.SAVE_DIR = generate_save_dir(
        "rmegan_checkpoints",
        config.SPLIT_STRATEGY,
        config.MODE,
        config.USE_FEATURE_MAPS,
        config.FIX_SAMPLES if config.MODE == "sparse" else None
    )
    
    print("\n⚙️  配置:")
    print(f"  数据路径: {config.DIR_MULTIBEAM}")
    print(f"  划分策略: {config.SPLIT_STRATEGY}")
    print(f"  采样模式: {config.MODE}")
    print(f"  使用Feature Map: {config.USE_FEATURE_MAPS}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  训练轮数: {config.NUM_EPOCHS}")
    print(f"  保存目录: {config.SAVE_DIR}")
    
    trainer = RMEGANTrainer(config)
    trainer.train()
    
    print("\n" + "=" * 80)
    print(f"✅ 训练结束! 最佳验证损失: {trainer.history['best_val']:.6f}")
    print(f"📁 模型保存在: {config.SAVE_DIR}")
    print("=" * 80)


def main_multi_sparsity():
    """多稀疏度实验主函数"""
    config = Config()
    config.NUM_EPOCHS = 15
    
    run_multi_sparsity_experiment(
        config, 
        sparsity_levels=[100, 200, 400, 819, 1000, 2000]
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "multi_sparsity":
        main_multi_sparsity()
    else:
        main()