"""
RME-GAN 模型评估 - Jupyter Notebook 版本
直接复制到Notebook单元格中运行

【修复版本】SSIM 计算已与 RadioUNet 统一：
1. 将无效区域设为 0（而非裁剪）
2. 动态计算 data_range（而非固定 1.0）

使用方法：
1. 复制全部代码到一个Cell中运行
2. 调用 run_evaluation() 函数进行评估
"""

# ============================================================================
# Cell 1: 导入库和配置
# ============================================================================

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
from tqdm.auto import tqdm  # Jupyter专用进度条
from datetime import datetime
import warnings
import re

warnings.filterwarnings("ignore")

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib未安装，可视化功能不可用")

# 尝试导入SSIM
try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠️ scikit-image未安装，使用简化SSIM计算")

print("✅ 库导入完成")

# ============================================================================
# Cell 2: 配置参数（根据实际情况修改）
# ============================================================================

# ⚠️ 请根据您的实际路径修改以下配置
DATA_CONFIG = {
    "DIR_MULTIBEAM": "Dataset/radiomaps",
    "DIR_HEIGHT_MAPS": "Dataset/height_maps",
    "DIR_FEATURE_MAPS": "Dataset/beam_maps",
}

# 实验目录
EXPERIMENTS_DIR = "Pretrained_Model/GAN"

# 8个实验的配置
EXPERIMENT_CONFIGS = {
    "random_dense_encoding": {
        "SPLIT_STRATEGY": "random", "MODE": "dense",
        "USE_FEATURE_MAPS": False, "WAVE_TYPE": None,
        "FIX_SAMPLES": None, "MODEL_INPUT_CH": 5
    },
    "random_dense_feature": {
        "SPLIT_STRATEGY": "random", "MODE": "dense",
        "USE_FEATURE_MAPS": True, "WAVE_TYPE": "spherical",
        "FIX_SAMPLES": None, "MODEL_INPUT_CH": 3
    },
    "scene_dense_encoding": {
        "SPLIT_STRATEGY": "scene", "MODE": "dense",
        "USE_FEATURE_MAPS": False, "WAVE_TYPE": None,
        "FIX_SAMPLES": None, "MODEL_INPUT_CH": 5
    },
    "scene_dense_feature": {
        "SPLIT_STRATEGY": "scene", "MODE": "dense",
        "USE_FEATURE_MAPS": True, "WAVE_TYPE": "spherical",
        "FIX_SAMPLES": None, "MODEL_INPUT_CH": 3
    },
    "beam_dense_encoding": {
        "SPLIT_STRATEGY": "beam", "MODE": "dense",
        "USE_FEATURE_MAPS": False, "WAVE_TYPE": None,
        "FIX_SAMPLES": None, "MODEL_INPUT_CH": 5
    },
    "beam_dense_feature": {
        "SPLIT_STRATEGY": "beam", "MODE": "dense",
        "USE_FEATURE_MAPS": True, "WAVE_TYPE": "spherical",
        "FIX_SAMPLES": None, "MODEL_INPUT_CH": 3
    },
    "random_sparse_feature_samples819": {
        "SPLIT_STRATEGY": "random", "MODE": "sparse",
        "USE_FEATURE_MAPS": True, "WAVE_TYPE": "spherical",
        "FIX_SAMPLES": 819, "MODEL_INPUT_CH": 4
    },
    "random_sparse_encoding_samples819": {
        "SPLIT_STRATEGY": "random", "MODE": "sparse",
        "USE_FEATURE_MAPS": False, "WAVE_TYPE": None,
        "FIX_SAMPLES": 819, "MODEL_INPUT_CH": 6
    }
}

print(f"✅ 配置完成")
print(f"   实验目录: {EXPERIMENTS_DIR}")
print(f"   数据目录: {DATA_CONFIG['DIR_MULTIBEAM']}")

# ============================================================================
# Cell 3: 评估指标计算函数
# ============================================================================

# dB域参数（与数据集保持一致）
DB_MIN = -300.0  # 最小dB值（NO_LABEL_VALUE）
DB_MAX = 0.0     # 最大dB值

def normalized_to_db(normalized_value):
    """
    将归一化值 [0, 1] 转换回 dB 域 [-300, 0]
    
    归一化公式: normalized = (db - DB_MIN) / (DB_MAX - DB_MIN)
    反归一化:   db = normalized * (DB_MAX - DB_MIN) + DB_MIN
    """
    return normalized_value * (DB_MAX - DB_MIN) + DB_MIN


def compute_metrics_db(pred_norm, target_norm, valid_mask):
    """
    在dB域计算MAE和RMSE（仅在有效区域）
    
    Args:
        pred_norm: 预测值，归一化 [0, 1]
        target_norm: 目标值，归一化 [0, 1]
        valid_mask: 有效区域掩码
    
    Returns:
        mae_db: dB域的MAE
        rmse_db: dB域的RMSE
    """
    # 转换到dB域
    pred_db = normalized_to_db(pred_norm)
    target_db = normalized_to_db(target_norm)
    
    # 仅在有效区域计算
    pred_valid = pred_db[valid_mask]
    target_valid = target_db[valid_mask]
    
    # MAE (dB)
    mae_db = np.abs(pred_valid - target_valid).mean()
    
    # RMSE (dB)
    mse_db = ((pred_valid - target_valid) ** 2).mean()
    rmse_db = np.sqrt(mse_db)
    
    return mae_db, rmse_db


def compute_ssim(pred, target, valid_mask=None, data_range=None):
    """
    计算SSIM（在归一化域，与RadioUNet保持一致）
    
    【修复】处理方式已与RadioUNet统一：
    1. 将无效区域设为0（而非裁剪）
    2. 动态计算data_range（有效区域内的max-min）
    
    Args:
        pred: 预测值，归一化 [0, 1]
        target: 目标值，归一化 [0, 1]
        valid_mask: 有效区域掩码
        data_range: 数据范围（如果为None则动态计算）
    
    Returns:
        ssim_value: SSIM值
    """
    # 将无效区域设为0（与RadioUNet一致）
    pred_masked = pred.copy()
    target_masked = target.copy()
    
    if valid_mask is not None:
        pred_masked[valid_mask == 0] = 0
        target_masked[valid_mask == 0] = 0
        
        # 动态计算data_range（与RadioUNet一致）
        if data_range is None:
            valid_target = target[valid_mask > 0]
            if len(valid_target) > 0:
                data_range = valid_target.max() - valid_target.min()
            else:
                return 0.0
    else:
        if data_range is None:
            data_range = target.max() - target.min()
    
    # 处理data_range为0的情况（与RadioUNet一致）
    if data_range <= 0:
        # 如果目标值全相同，检查预测是否也相同
        if valid_mask is not None:
            pred_valid = pred[valid_mask > 0]
            target_valid = target[valid_mask > 0]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        mae = np.mean(np.abs(pred_valid - target_valid))
        return 1.0 if mae == 0 else 0.0
    
    # 使用 skimage 计算 SSIM
    if HAS_SKIMAGE:
        try:
            return skimage_ssim(target_masked, pred_masked, data_range=data_range)
        except Exception:
            pass
    
    # 简化SSIM（备用方案）
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu1, mu2 = np.mean(pred_masked), np.mean(target_masked)
    sigma1_sq, sigma2_sq = np.var(pred_masked), np.var(target_masked)
    sigma12 = np.cov(pred_masked.flatten(), target_masked.flatten())[0, 1]
    return ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))


print("✅ 评估指标函数定义完成")
print(f"   dB域范围: [{DB_MIN}, {DB_MAX}] dB")
print(f"   【修复】SSIM计算已与RadioUNet统一")

# ============================================================================
# Cell 4: 网络结构定义
# ============================================================================

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
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(nc*4, use_dropout) for _ in range(num_res_blocks)])
        
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
        return self.out(d1)

print("✅ 网络结构定义完成")

# ============================================================================
# Cell 5: 评估器类
# ============================================================================

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, exp_dir=EXPERIMENTS_DIR, data_config=DATA_CONFIG):
        self.exp_dir = exp_dir
        self.data_config = data_config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*60}")
        print(f"📊 RME-GAN 评估器初始化")
        print(f"   设备: {self.device}")
        print(f"   实验目录: {exp_dir}")
        print(f"   【修复】SSIM计算与RadioUNet统一")
        print(f"{'='*60}")
        
        # 查找实验
        self.experiments = self._find_experiments()
        print(f"\n找到 {len(self.experiments)} 个实验:")
        for exp in self.experiments:
            print(f"  ✓ {exp}")
    
    def _find_experiments(self):
        """查找所有实验"""
        experiments = []
        if not os.path.exists(self.exp_dir):
            print(f"⚠️ 实验目录不存在: {self.exp_dir}")
            return experiments
        
        for item in os.listdir(self.exp_dir):
            item_path = os.path.join(self.exp_dir, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "best_G.pth")):
                    experiments.append(item)
        return sorted(experiments)
    
    def _get_config(self, exp_name):
        """获取实验配置"""
        # 从预定义配置获取
        if exp_name in EXPERIMENT_CONFIGS:
            return EXPERIMENT_CONFIGS[exp_name]
        
        # 从config.json读取
        config_path = os.path.join(self.exp_dir, exp_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # 从目录名解析
        config = {
            'SPLIT_STRATEGY': 'random', 'MODE': 'dense',
            'USE_FEATURE_MAPS': True, 'WAVE_TYPE': 'spherical',
            'FIX_SAMPLES': None, 'MODEL_INPUT_CH': 3
        }
        
        if 'beam' in exp_name:
            config['SPLIT_STRATEGY'] = 'beam'
        elif 'scene' in exp_name:
            config['SPLIT_STRATEGY'] = 'scene'
        
        if 'sparse' in exp_name:
            config['MODE'] = 'sparse'
            match = re.search(r'samples(\d+)', exp_name)
            if match:
                config['FIX_SAMPLES'] = int(match.group(1))
        
        if 'encoding' in exp_name:
            config['USE_FEATURE_MAPS'] = False
            config['WAVE_TYPE'] = None
            config['MODEL_INPUT_CH'] = 5 if config['MODE'] == 'dense' else 6
        else:
            config['WAVE_TYPE'] = 'plane' if 'plane' in exp_name else 'spherical'
            config['MODEL_INPUT_CH'] = 3 if config['MODE'] == 'dense' else 4
        
        return config
    
    def _load_model(self, exp_name, config):
        """加载模型"""
        model_path = os.path.join(self.exp_dir, exp_name, "best_G.pth")
        model = RMEGANGenerator(input_channels=config['MODEL_INPUT_CH']).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def _create_test_loader(self, config):
        """创建测试数据加载器"""
        try:
            from multiconfig_dataset_prepcocess_GAN import RMEGANDataset
        except ImportError:
            print("❌ 无法导入 multiconfig_dataset_prepcocess_GAN")
            print("   请确保该文件在当前目录或Python路径中")
            return None
        
        dataset = RMEGANDataset(
            phase="test",
            dir_multibeam=self.data_config["DIR_MULTIBEAM"],
            dir_height_maps=self.data_config["DIR_HEIGHT_MAPS"],
            dir_feature_maps=self.data_config["DIR_FEATURE_MAPS"],
            split_strategy=config['SPLIT_STRATEGY'],
            train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
            mode=config['MODE'],
            fix_samples=config.get('FIX_SAMPLES', 819) or 819,
            use_feature_maps=config['USE_FEATURE_MAPS'],
            wave_type=config.get('WAVE_TYPE', 'spherical') or 'spherical',
            random_seed=42
        )
        
        return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    def _prepare_inputs(self, batch, config):
        """准备模型输入"""
        mode = config['MODE']
        use_feature_maps = config['USE_FEATURE_MAPS']
        
        if mode == "dense":
            inputs, targets, valids = batch
            samples_mask = None
        else:
            inputs, targets, samples_mask, valids = batch
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        valids = valids.to(self.device)
        
        if use_feature_maps:
            condition_input = inputs[:, [1, 2, 3], :, :]
        else:
            condition_input = inputs[:, [1, 2, 3, 4, 6], :, :]
        
        if mode == "sparse":
            samples_mask = samples_mask.to(self.device)
            sampled_input = samples_mask * targets
            inputs = torch.cat([sampled_input, condition_input], dim=1)
        else:
            inputs = condition_input
        
        return inputs, targets, valids, samples_mask
    
    def evaluate_single(self, exp_name, visualize=False, num_vis_samples=5):
        """评估单个实验"""
        print(f"\n{'='*60}")
        print(f"📊 评估: {exp_name}")
        print(f"{'='*60}")
        
        config = self._get_config(exp_name)
        print(f"  划分策略: {config['SPLIT_STRATEGY']}")
        print(f"  模式: {config['MODE']}")
        print(f"  Feature Map: {config['USE_FEATURE_MAPS']}")
        if config['USE_FEATURE_MAPS']:
            print(f"  波类型: {config['WAVE_TYPE']}")
        
        # 加载模型
        model = self._load_model(exp_name, config)
        print(f"  ✅ 模型已加载 (输入通道: {config['MODEL_INPUT_CH']})")
        
        # 创建数据加载器
        test_loader = self._create_test_loader(config)
        if test_loader is None:
            return None
        print(f"  ✅ 测试集: {len(test_loader.dataset)} 样本")
        
        # 评估（dB域）
        total_mae_db, total_mse_db, total_ssim = 0.0, 0.0, 0.0
        n_samples = 0
        
        # 保存用于可视化的数据
        vis_data = {'pred': [], 'target': [], 'valid': []}
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="评估中")
            for batch in pbar:
                inputs, targets, valids, _ = self._prepare_inputs(batch, config)
                
                outputs = model(inputs)
                outputs_norm = (outputs + 1) / 2  # 归一化到[0,1]
                
                outputs_np = outputs_norm.cpu().numpy()
                targets_np = targets.cpu().numpy()
                valids_np = valids.cpu().numpy()
                
                for i in range(outputs_np.shape[0]):
                    pred = outputs_np[i, 0]
                    target = targets_np[i, 0]
                    valid = valids_np[i] > 0.5
                    n_valid = valid.sum()
                    
                    if n_valid == 0:
                        continue
                    
                    # 保存可视化数据
                    if len(vis_data['pred']) < num_vis_samples:
                        vis_data['pred'].append(pred.copy())
                        vis_data['target'].append(target.copy())
                        vis_data['valid'].append(valid.copy())
                    
                    # 在dB域计算MAE和RMSE
                    mae_db, rmse_db = compute_metrics_db(pred, target, valid)
                    
                    # SSIM（在归一化域计算，与RadioUNet统一）
                    ssim_val = compute_ssim(pred, target, valid)  # 不再传固定data_range
                    
                    total_mae_db += mae_db
                    total_mse_db += rmse_db ** 2  # 累积平方以便后续计算整体RMSE
                    total_ssim += ssim_val
                    n_samples += 1
                
                if n_samples > 0:
                    pbar.set_postfix({
                        'MAE(dB)': f'{total_mae_db/n_samples:.2f}',
                        'SSIM': f'{total_ssim/n_samples:.4f}'
                    })
        
        if n_samples == 0:
            print("  ❌ 没有有效样本")
            return None
        
        metrics = {
            'MAE_dB': float(total_mae_db / n_samples),
            'RMSE_dB': float(np.sqrt(total_mse_db / n_samples)),
            'SSIM': float(total_ssim / n_samples),
            'n_samples': n_samples
        }
        
        print(f"\n  📊 评估结果 (dB域，仅有效区域):")
        print(f"     MAE:  {metrics['MAE_dB']:.4f} dB")
        print(f"     RMSE: {metrics['RMSE_dB']:.4f} dB")
        print(f"     SSIM: {metrics['SSIM']:.6f}")
        print(f"     样本数: {n_samples}")
        
        # 保存结果
        result_path = os.path.join(self.exp_dir, exp_name, "eval_results.json")
        with open(result_path, 'w') as f:
            json.dump({
                'config': config, 
                'metrics': metrics,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'note': 'SSIM calculation unified with RadioUNet'
            }, f, indent=2)
        print(f"  💾 结果已保存: {result_path}")
        
        # 可视化
        if visualize and HAS_MATPLOTLIB:
            self._visualize(exp_name, vis_data, num_vis_samples)
        
        return metrics
    
    def _visualize(self, exp_name, vis_data, num_samples):
        """生成可视化"""
        n = min(num_samples, len(vis_data['pred']))
        if n == 0:
            return
        
        fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n):
            pred = vis_data['pred'][i]
            target = vis_data['target'][i]
            valid = vis_data['valid'][i]
            
            # 转换到dB域计算误差
            pred_db = normalized_to_db(pred)
            target_db = normalized_to_db(target)
            error_db = np.abs(pred_db - target_db)
            
            # Ground Truth (dB)
            im0 = axes[i, 0].imshow(target_db, cmap='viridis', vmin=-150, vmax=0)
            axes[i, 0].set_title(f'Ground Truth #{i+1} (dB)')
            axes[i, 0].axis('off')
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Prediction (dB)
            im1 = axes[i, 1].imshow(pred_db, cmap='viridis', vmin=-150, vmax=0)
            axes[i, 1].set_title(f'Prediction #{i+1} (dB)')
            axes[i, 1].axis('off')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Error Map (dB)
            mae_db_i = error_db[valid].mean() if valid.sum() > 0 else 0
            error_masked = np.where(valid, error_db, np.nan)
            im2 = axes[i, 2].imshow(error_masked, cmap='hot', vmin=0, vmax=30)
            axes[i, 2].set_title(f'Error (MAE={mae_db_i:.2f} dB)')
            axes[i, 2].axis('off')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # Valid Mask
            axes[i, 3].imshow(valid, cmap='gray')
            axes[i, 3].set_title('Valid Mask')
            axes[i, 3].axis('off')
        
        plt.suptitle(f'Evaluation: {exp_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        vis_path = os.path.join(self.exp_dir, exp_name, "eval_visualization.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  ✅ 可视化已保存: {vis_path}")
    
    def evaluate_all(self, visualize=False):
        """评估所有实验"""
        all_results = {}
        
        for exp_name in self.experiments:
            try:
                metrics = self.evaluate_single(exp_name, visualize=visualize)
                if metrics:
                    all_results[exp_name] = metrics
            except Exception as e:
                print(f"  ❌ 评估失败: {e}")
                import traceback
                traceback.print_exc()
                all_results[exp_name] = {'error': str(e)}
        
        # 打印汇总
        self._print_summary(all_results)
        
        # 保存汇总
        self._save_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results):
        """打印汇总表格"""
        print("\n" + "=" * 95)
        print("📊 评估汇总 (dB域，仅有效区域，SSIM与RadioUNet统一)")
        print("=" * 95)
        print(f"{'实验名称':<45} {'MAE(dB)':<12} {'RMSE(dB)':<12} {'SSIM':<12}")
        print("-" * 95)
        
        # 按MAE排序
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if 'error' not in v],
            key=lambda x: x[1]['MAE_dB']
        )
        
        for exp_name, metrics in sorted_results:
            print(f"{exp_name:<45} {metrics['MAE_dB']:.4f}       {metrics['RMSE_dB']:.4f}       {metrics['SSIM']:.6f}")
        
        # 打印错误
        for exp_name, metrics in results.items():
            if 'error' in metrics:
                print(f"{exp_name:<45} ❌ ERROR")
        
        print("-" * 95)
        
        if sorted_results:
            best = sorted_results[0]
            print(f"\n🏆 最佳模型: {best[0]}")
            print(f"   MAE={best[1]['MAE_dB']:.4f} dB, RMSE={best[1]['RMSE_dB']:.4f} dB, SSIM={best[1]['SSIM']:.6f}")
    
    def _save_summary(self, results):
        """保存汇总"""
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'note': 'MAE and RMSE are in dB domain, computed only on valid regions. SSIM calculation unified with RadioUNet.',
            'experiments': results
        }
        
        # JSON
        json_path = os.path.join(self.exp_dir, "evaluation_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # CSV
        csv_path = os.path.join(self.exp_dir, "evaluation_summary.csv")
        with open(csv_path, 'w') as f:
            f.write("Experiment,MAE_dB,RMSE_dB,SSIM,Samples\n")
            for exp_name, metrics in results.items():
                if 'error' not in metrics:
                    f.write(f"{exp_name},{metrics['MAE_dB']:.4f},{metrics['RMSE_dB']:.4f},{metrics['SSIM']:.6f},{metrics['n_samples']}\n")
        
        print(f"\n💾 汇总已保存:")
        print(f"   JSON: {json_path}")
        print(f"   CSV:  {csv_path}")

print("✅ 评估器类定义完成")

# ============================================================================
# Cell 6: 便捷函数
# ============================================================================

def run_evaluation(exp_dir=EXPERIMENTS_DIR, single=None, visualize=False):
    """
    运行评估的便捷函数
    
    参数:
        exp_dir: 实验目录路径
        single: 单个实验名称（None表示评估所有）
        visualize: 是否生成可视化
    
    返回:
        评估结果字典
    
    使用示例:
        # 评估所有实验
        results = run_evaluation()
        
        # 评估单个实验
        results = run_evaluation(single='random_dense_feature')
        
        # 带可视化
        results = run_evaluation(visualize=True)
    """
    evaluator = ModelEvaluator(exp_dir=exp_dir)
    
    if single:
        return evaluator.evaluate_single(single, visualize=visualize)
    else:
        return evaluator.evaluate_all(visualize=visualize)

print("✅ 便捷函数定义完成")
print("\n" + "=" * 60)
print("🎉 所有代码加载完成！")
print("=" * 60)
print("\n【修复内容】SSIM计算已与RadioUNet统一:")
print("  1. 无效区域设为0（而非裁剪）")
print("  2. 动态计算data_range（而非固定1.0）")
print("\n使用方法:")
print("  # 评估所有实验")
print("  results = run_evaluation()")
print("")
print("  # 评估单个实验")
print("  results = run_evaluation(single='random_dense_feature')")
print("")
print("  # 带可视化")
print("  results = run_evaluation(visualize=True)")
print("=" * 60)

run_evaluation()