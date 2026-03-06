"""
完整的测试集评估脚本 - dB域版本
包含：MAE、MSE、RMSE、SSIM评估 + 可视化
注意：
1. 所有指标在dB域计算（而非归一化后的[0,1]域）
2. 排除建筑物和无效点区域
3. SSIM也只在有效区域计算
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# CUDA配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 导入自定义模块
from multiconfig_dataset_prepcocess_Unet import MultiBeamRadioDataset
from modules_Unet import RadioWNet


# ==================== 评估配置 ====================
class EvalConfig:
    """评估配置类"""
    
    # 数据路径
    DIR_MULTIBEAM = "Dataset/radiomaps"
    DIR_HEIGHT_MAPS = "Dataset/height_maps"
    DIR_FEATURE_MAPS = "Dataset/beam_maps"
    
    # 测试参数
    RANDOM_SEED = 42
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    BATCH_SIZE = 64
    
    # 归一化参数（从数据集代码中获取）
    NO_LABEL_VALUE = -300  # dB
    MAX_DB_VALUE = 0  # dB
    
    # 保存路径
    SAVE_DIR = "My_Net"
    RESULTS_DIR = "evaluation_results"
    
    # 模型列表配置
    MODELS = [
        {
            "name": "Solution1_environment",
            "path": "Pretrained_Model/Unet/Solution1_dense_seed42_environment_Second_Net.pt",
            "task_mode": "dense",
            "use_feature_maps": True,
            "input_channels": [1, 2],  # Tx, Height
            "split_strategy": "random",
            "feature_map_dir": DIR_FEATURE_MAPS,
        },
        {
            "name": "Solution1_featuremap",
            "path": "Pretrained_Model/Unet/Solution1_dense_seed42_featuremap_Second_Net.pt",
            "task_mode": "dense",
            "use_feature_maps": True,
            "input_channels": [1, 2, 3],  # Tx, Height, FeatureMap
            "split_strategy": "random",
            "feature_map_dir": DIR_FEATURE_MAPS,
        },
        {
            "name": "Solution1_continuous",
            "path": "Pretrained_Model/Unet/Solution1_dense_seed42_Second_Net.pt",
            "task_mode": "dense",
            "use_feature_maps": False,
            "input_channels": [1, 2, 3, 4, 6],  # Tx, Height, Freq, TR_log, BeamID
            "split_strategy": "random",
            "feature_map_dir": None,
        },
        {
            "name": "Solution2_sparse_featuremap",
            "path": "Pretrained_Model/Unet/Solution2_sparse_seed42_featuremap_Second_Net.pt",
            "task_mode": "sparse",
            "use_feature_maps": True,
            "input_channels": [1, 2, 3],  # 加上samples*targets作为第0通道
            "split_strategy": "random",
            "feature_map_dir": DIR_FEATURE_MAPS,
            "use_sparse_input": True,
        },
        {
            "name": "Solution2_sparse_continuous",
            "path": "Pretrained_Model/Unet/Solution2_sparse_seed42_Second_Net.pt",
            "task_mode": "sparse",
            "use_feature_maps": False,
            "input_channels": [1, 2, 3, 4, 6],  # 加上samples*targets作为第0通道
            "split_strategy": "random",
            "feature_map_dir": None,
            "use_sparse_input": True,
        },
        {
            "name": "Solution3_1_beam_featuremap",
            "path": "Pretrained_Model/Unet/Solution3_1_dense_seed42_featuremap_Second_Net.pt",
            "task_mode": "dense",
            "use_feature_maps": True,
            "input_channels": [1, 2, 3],
            "split_strategy": "beam",
            "feature_map_dir": DIR_FEATURE_MAPS,
        },
        {
            "name": "Solution3_1_beam_continuous",
            "path": "Pretrained_Model/Unet/Solution3_1_dense_seed42_Second_Net.pt",
            "task_mode": "dense",
            "use_feature_maps": False,
            "input_channels": [1, 2, 3, 4, 6],
            "split_strategy": "beam",
            "feature_map_dir": None,
        },
        {
            "name": "Solution3_2_scene_featuremap",
            "path": "Pretrained_Model/Unet/Solution3_2_dense_seed42_featuremap_Second_Net.pt",
            "task_mode": "dense",
            "use_feature_maps": True,
            "input_channels": [1, 2, 3],
            "split_strategy": "scene",
            "feature_map_dir": DIR_FEATURE_MAPS,
        },
        {
            "name": "Solution3_2_scene_continuous",
            "path": "Pretrained_Model/Unet/Solution3_2_dense_seed42_Second_Net.pt",
            "task_mode": "dense",
            "use_feature_maps": False,
            "input_channels": [1, 2, 3, 4, 6],
            "split_strategy": "scene",
            "feature_map_dir": None,
        },
    ]


# ==================== 归一化转换函数 ====================
def denormalize_to_dB(normalized_value, NO_LABEL_VALUE=-300, MAX_DB_VALUE=0):
    """
    将归一化的[0,1]值转换回dB域
    
    归一化公式（数据集中）：
    normalized = (dB - NO_LABEL_VALUE) / (MAX_DB_VALUE - NO_LABEL_VALUE)
    normalized = (dB - (-300)) / (0 - (-300))
    normalized = (dB + 300) / 300
    
    反归一化公式：
    dB = normalized * (MAX_DB_VALUE - NO_LABEL_VALUE) + NO_LABEL_VALUE
    dB = normalized * 300 - 300
    
    Args:
        normalized_value: 归一化后的值 [0, 1]
        NO_LABEL_VALUE: 最小dB值（默认-300）
        MAX_DB_VALUE: 最大dB值（默认0）
    
    Returns:
        dB值 [NO_LABEL_VALUE, MAX_DB_VALUE]
    """
    dB_value = normalized_value * (MAX_DB_VALUE - NO_LABEL_VALUE) + NO_LABEL_VALUE
    return dB_value


# ==================== 只在有效区域计算SSIM ====================
def calculate_ssim_valid_region(pred, target, valid_mask):
    """
    只在有效区域计算SSIM
    
    方法：提取有效区域的最小外接矩形，只在该区域计算SSIM
    同时使用full=True获取SSIM map，然后只对有效像素取平均
    
    Args:
        pred: (H, W) 预测值
        target: (H, W) 真实值
        valid_mask: (H, W) 有效区域掩码 (1=有效, 0=无效)
    
    Returns:
        float: 有效区域的SSIM值
    """
    # 确保是numpy数组
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    if torch.is_tensor(valid_mask):
        valid_mask = valid_mask.cpu().numpy()
    
    # 检查有效像素数量
    num_valid = np.sum(valid_mask > 0)
    if num_valid == 0:
        return float('nan')
    
    # 方法1: 使用SSIM map，只对有效像素取平均
    # 计算data_range（只考虑有效区域）
    target_valid_values = target[valid_mask > 0]
    pred_valid_values = pred[valid_mask > 0]
    
    # 合并计算data_range，确保覆盖两者的范围
    all_valid_values = np.concatenate([target_valid_values, pred_valid_values])
    data_range = all_valid_values.max() - all_valid_values.min()
    
    if data_range == 0:
        # 如果所有值相同，检查预测是否也相同
        if np.allclose(pred_valid_values, target_valid_values):
            return 1.0
        else:
            return 0.0
    
    # 计算完整的SSIM map
    # win_size必须是奇数且不超过图像最小维度
    min_dim = min(pred.shape)
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3
    
    try:
        # 获取SSIM map
        ssim_value, ssim_map = ssim(
            target, pred, 
            data_range=data_range,
            win_size=win_size,
            full=True
        )
        
        # 只对有效区域的SSIM值取平均
        # 注意：SSIM map边缘会有一些无效值（因为滑动窗口），需要处理
        ssim_map_valid = ssim_map[valid_mask > 0]
        
        # 排除可能的nan值
        ssim_map_valid = ssim_map_valid[~np.isnan(ssim_map_valid)]
        
        if len(ssim_map_valid) == 0:
            return float('nan')
        
        return np.mean(ssim_map_valid)
        
    except Exception as e:
        print(f"    SSIM计算警告: {e}")
        return float('nan')


def calculate_ssim_valid_region_v2(pred, target, valid_mask):
    """
    只在有效区域计算SSIM - 方法2：提取有效区域的bounding box
    
    Args:
        pred: (H, W) 预测值
        target: (H, W) 真实值
        valid_mask: (H, W) 有效区域掩码 (1=有效, 0=无效)
    
    Returns:
        float: 有效区域的SSIM值
    """
    # 确保是numpy数组
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    if torch.is_tensor(valid_mask):
        valid_mask = valid_mask.cpu().numpy()
    
    # 找到有效区域的bounding box
    valid_indices = np.where(valid_mask > 0)
    if len(valid_indices[0]) == 0:
        return float('nan')
    
    row_min, row_max = valid_indices[0].min(), valid_indices[0].max()
    col_min, col_max = valid_indices[1].min(), valid_indices[1].max()
    
    # 提取bounding box区域
    pred_crop = pred[row_min:row_max+1, col_min:col_max+1]
    target_crop = target[row_min:row_max+1, col_min:col_max+1]
    mask_crop = valid_mask[row_min:row_max+1, col_min:col_max+1]
    
    # 将无效区域设为该区域的均值（减少边界影响）
    mean_val = target_crop[mask_crop > 0].mean()
    pred_crop_masked = pred_crop.copy()
    target_crop_masked = target_crop.copy()
    pred_crop_masked[mask_crop == 0] = mean_val
    target_crop_masked[mask_crop == 0] = mean_val
    
    # 计算data_range
    data_range = target_crop[mask_crop > 0].max() - target_crop[mask_crop > 0].min()
    
    if data_range == 0:
        pred_valid = pred_crop[mask_crop > 0]
        target_valid = target_crop[mask_crop > 0]
        if np.allclose(pred_valid, target_valid):
            return 1.0
        else:
            return 0.0
    
    # 确保crop区域足够大
    min_dim = min(pred_crop_masked.shape)
    if min_dim < 7:
        # 区域太小，使用简单的相关系数代替
        pred_valid = pred_crop[mask_crop > 0]
        target_valid = target_crop[mask_crop > 0]
        if len(pred_valid) < 2:
            return float('nan')
        corr = np.corrcoef(pred_valid, target_valid)[0, 1]
        return max(0, corr)  # SSIM范围是[-1,1]，但通常为正
    
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    
    try:
        ssim_value = ssim(
            target_crop_masked, pred_crop_masked,
            data_range=data_range,
            win_size=win_size
        )
        return ssim_value
    except Exception as e:
        print(f"    SSIM计算警告: {e}")
        return float('nan')


# ==================== 评估指标计算（dB域）====================
def calculate_metrics_dB(pred_normalized, target_normalized, valid_mask, 
                        NO_LABEL_VALUE=-300, MAX_DB_VALUE=0):
    """
    计算评估指标（在dB域，排除建筑物和无效区域）
    
    Args:
        pred_normalized: (H, W) 预测值（归一化后，[0,1]）
        target_normalized: (H, W) 真实值（归一化后，[0,1]）
        valid_mask: (H, W) 有效区域掩码
        NO_LABEL_VALUE: 最小dB值
        MAX_DB_VALUE: 最大dB值
    
    Returns:
        dict: {'mae_dB': float, 'mse_dB': float, 'rmse_dB': float, 'ssim': float}
    """
    # 确保是numpy数组
    if torch.is_tensor(pred_normalized):
        pred_normalized = pred_normalized.cpu().numpy()
    if torch.is_tensor(target_normalized):
        target_normalized = target_normalized.cpu().numpy()
    if torch.is_tensor(valid_mask):
        valid_mask = valid_mask.cpu().numpy()
    
    # ⭐ 转换到dB域
    pred_dB = denormalize_to_dB(pred_normalized, NO_LABEL_VALUE, MAX_DB_VALUE)
    target_dB = denormalize_to_dB(target_normalized, NO_LABEL_VALUE, MAX_DB_VALUE)
    
    # 提取有效区域
    pred_valid = pred_dB[valid_mask > 0]
    target_valid = target_dB[valid_mask > 0]
    
    if len(pred_valid) == 0:
        return {
            'mae_dB': float('nan'),
            'mse_dB': float('nan'),
            'rmse_dB': float('nan'),
            'ssim': float('nan'),
            'num_valid_pixels': 0
        }
    
    # ⭐ 在dB域计算MAE
    mae_dB = np.mean(np.abs(pred_valid - target_valid))
    
    # ⭐ 在dB域计算MSE和RMSE
    mse_dB = np.mean((pred_valid - target_valid) ** 2)
    rmse_dB = np.sqrt(mse_dB)
    
    # ⭐ SSIM - 只在有效区域计算（使用SSIM map方法）
    # 在归一化域计算SSIM，因为SSIM对值的绝对大小不敏感
    ssim_value = calculate_ssim_valid_region(pred_normalized, target_normalized, valid_mask)
    
    return {
        'mae_dB': mae_dB,
        'mse_dB': mse_dB,
        'rmse_dB': rmse_dB,
        'ssim': ssim_value,
        'num_valid_pixels': len(pred_valid)
    }


# ==================== 模型评估 ====================
def evaluate_model(model, dataloader, device, model_config, NO_LABEL_VALUE=-300, MAX_DB_VALUE=0):
    """评估单个模型"""
    model.eval()
    
    all_metrics = []
    all_predictions_norm = []
    all_targets_norm = []
    all_predictions_dB = []
    all_targets_dB = []
    all_masks = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # 根据task_mode解包数据
            if model_config['task_mode'] == 'dense':
                inputs, targets, vamask = batch_data
            else:  # sparse
                inputs, targets, samples, vamask = batch_data
            
            # 准备输入
            if model_config.get('use_sparse_input', False):
                # Solution2: 稀疏输入 (samples*targets + 其他通道)
                inputs1 = samples * targets
                selected_channels = inputs[:, model_config['input_channels'], :, :]
                combined = torch.cat([inputs1, selected_channels], dim=1)
                model_input = combined.to(device)
            else:
                # 普通输入：选择指定通道
                model_input = inputs[:, model_config['input_channels'], :, :].to(device)
            
            targets = targets.to(device)
            vamask = vamask.to(device)
            
            # 前向传播
            outputs1, outputs2 = model(model_input)
            pred = outputs2  # 使用第二个U-Net的输出
            
            # 逐样本计算指标
            batch_size = pred.shape[0]
            for i in range(batch_size):
                pred_i = pred[i, 0]  # (H, W) 归一化值
                target_i = targets[i, 0]  # (H, W) 归一化值
                mask_i = vamask[i]  # (H, W)
                
                # ⭐ 在dB域计算指标
                metrics = calculate_metrics_dB(
                    pred_i, target_i, mask_i, 
                    NO_LABEL_VALUE, MAX_DB_VALUE
                )
                all_metrics.append(metrics)
                
                # 保存部分样本用于可视化
                if batch_idx == 0 and i < 5:
                    pred_np = pred_i.cpu().numpy()
                    target_np = target_i.cpu().numpy()
                    mask_np = mask_i.cpu().numpy()
                    
                    # 保存归一化版本
                    all_predictions_norm.append(pred_np)
                    all_targets_norm.append(target_np)
                    all_masks.append(mask_np)
                    
                    # 保存dB版本
                    pred_dB = denormalize_to_dB(pred_np, NO_LABEL_VALUE, MAX_DB_VALUE)
                    target_dB = denormalize_to_dB(target_np, NO_LABEL_VALUE, MAX_DB_VALUE)
                    all_predictions_dB.append(pred_dB)
                    all_targets_dB.append(target_dB)
            
            if batch_idx % 10 == 0:
                print(f"  处理批次 {batch_idx}/{len(dataloader)}")
    
    # 汇总结果
    valid_metrics = [m for m in all_metrics if not np.isnan(m['mae_dB'])]
    
    # 对SSIM单独处理nan
    valid_ssim = [m['ssim'] for m in valid_metrics if not np.isnan(m['ssim'])]
    
    summary = {
        'mae_dB_mean': np.mean([m['mae_dB'] for m in valid_metrics]),
        'mae_dB_std': np.std([m['mae_dB'] for m in valid_metrics]),
        'mse_dB_mean': np.mean([m['mse_dB'] for m in valid_metrics]),
        'mse_dB_std': np.std([m['mse_dB'] for m in valid_metrics]),
        'rmse_dB_mean': np.sqrt(np.mean([m['mse_dB'] for m in valid_metrics])),
        'rmse_dB_std': np.std([m['rmse_dB'] for m in valid_metrics]),
        'ssim_mean': np.mean(valid_ssim) if valid_ssim else float('nan'),
        'ssim_std': np.std(valid_ssim) if valid_ssim else float('nan'),
        'num_samples': len(valid_metrics),
        'num_valid_ssim_samples': len(valid_ssim),
    }
    
    return summary, all_predictions_norm, all_targets_norm, all_predictions_dB, all_targets_dB, all_masks


# ==================== 可视化 ====================
def visualize_predictions(predictions_norm, targets_norm, predictions_dB, targets_dB, 
                         masks, model_name, save_dir):
    """
    可视化预测结果
    
    上半部分：归一化域（用于模型训练）
    下半部分：dB域（用于实际评估）
    """
    num_samples = min(5, len(predictions_norm))
    
    # 创建两行图：归一化域 + dB域
    fig, axes = plt.subplots(num_samples*2, 4, figsize=(16, 8*num_samples))
    
    for i in range(num_samples):
        # ===== 第一行：归一化域 =====
        row_idx = i * 2
        
        pred_norm = predictions_norm[i]
        target_norm = targets_norm[i]
        mask = masks[i]
        
        # 归一化域误差
        error_norm = np.abs(pred_norm - target_norm)
        error_norm[mask == 0] = 0
        
        # 目标（归一化）
        im0 = axes[row_idx, 0].imshow(target_norm, cmap='jet', vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f'Sample {i+1}: Target (Normalized)')
        axes[row_idx, 0].axis('off')
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046)
        
        # 预测（归一化）
        im1 = axes[row_idx, 1].imshow(pred_norm, cmap='jet', vmin=0, vmax=1)
        axes[row_idx, 1].set_title('Prediction (Normalized)')
        axes[row_idx, 1].axis('off')
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046)
        
        # 误差（归一化）
        im2 = axes[row_idx, 2].imshow(error_norm, cmap='hot', vmin=0, vmax=0.5)
        axes[row_idx, 2].set_title('Absolute Error (Normalized)')
        axes[row_idx, 2].axis('off')
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046)
        
        # 有效掩码
        im3 = axes[row_idx, 3].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 3].set_title('Valid Mask')
        axes[row_idx, 3].axis('off')
        plt.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046)
        
        # ===== 第二行：dB域 =====
        row_idx = i * 2 + 1
        
        pred_dB = predictions_dB[i]
        target_dB = targets_dB[i]
        
        # dB域误差
        error_dB = np.abs(pred_dB - target_dB)
        error_dB[mask == 0] = 0
        
        # 目标（dB）
        im0 = axes[row_idx, 0].imshow(target_dB, cmap='jet', vmin=-300, vmax=0)
        axes[row_idx, 0].set_title('Target (dB)')
        axes[row_idx, 0].axis('off')
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046)
        
        # 预测（dB）
        im1 = axes[row_idx, 1].imshow(pred_dB, cmap='jet', vmin=-300, vmax=0)
        axes[row_idx, 1].set_title('Prediction (dB)')
        axes[row_idx, 1].axis('off')
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046)
        
        # 误差（dB）- 使用更合理的范围
        im2 = axes[row_idx, 2].imshow(error_dB, cmap='hot', vmin=0, vmax=50)
        axes[row_idx, 2].set_title('Absolute Error (dB)')
        axes[row_idx, 2].axis('off')
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046)
        
        # 计算该样本的dB域指标
        pred_valid = pred_dB[mask > 0]
        target_valid = target_dB[mask > 0]
        mae_sample = np.mean(np.abs(pred_valid - target_valid))
        
        # 计算该样本的SSIM（只在有效区域）
        ssim_sample = calculate_ssim_valid_region(predictions_norm[i], targets_norm[i], mask)
        
        axes[row_idx, 3].text(0.5, 0.5, 
                             f'MAE: {mae_sample:.2f} dB\n' +
                             f'SSIM: {ssim_sample:.4f}\n' +
                             f'Min Error: {error_dB[mask > 0].min():.2f} dB\n' +
                             f'Max Error: {error_dB[mask > 0].max():.2f} dB\n' +
                             f'Valid Pixels: {np.sum(mask > 0)}',
                             ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[row_idx, 3].axis('off')
        axes[row_idx, 3].set_title('Metrics (dB)')
    
    plt.suptitle(f'{model_name} - Predictions on Test Set\n(Top: Normalized, Bottom: dB domain)', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{model_name}_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存可视化: {save_path}")
    plt.close()


# ==================== 结果汇总 ====================
def create_summary_table(all_results, save_dir):
    """创建结果汇总表格（dB域）"""
    
    # 准备数据
    data = []
    for model_name, result in all_results.items():
        data.append({
            'Model': model_name,
            'MAE (dB)': f"{result['mae_dB_mean']:.4f} ± {result['mae_dB_std']:.4f}",
            'MSE (dB²)': f"{result['mse_dB_mean']:.4f} ± {result['mse_dB_std']:.4f}",
            'RMSE (dB)': f"{result['rmse_dB_mean']:.4f} ± {result['rmse_dB_std']:.4f}",
            'SSIM': f"{result['ssim_mean']:.6f} ± {result['ssim_std']:.6f}",
            'Samples': result['num_samples'],
            'SSIM_Samples': result.get('num_valid_ssim_samples', result['num_samples'])
        })
    
    df = pd.DataFrame(data)
    
    # 保存为CSV
    csv_path = os.path.join(save_dir, 'evaluation_summary_dB.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 保存CSV: {csv_path}")
    
    # 打印表格
    print("\n" + "="*120)
    print("评估结果汇总（dB域，SSIM只在有效区域计算）")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    
    return df


def create_comparison_plots(all_results, save_dir):
    """创建对比图表（dB域）"""
    
    # 提取数据
    models = []
    mae_means = []
    mae_stds = []
    mse_means = []
    rmse_means = []
    ssim_means = []
    
    for model_name, result in all_results.items():
        models.append(model_name)
        mae_means.append(result['mae_dB_mean'])
        mae_stds.append(result['mae_dB_std'])
        mse_means.append(result['mse_dB_mean'])
        rmse_means.append(result['rmse_dB_mean'])
        ssim_means.append(result['ssim_mean'])
    
    # 设置颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # MAE对比（dB）
    axes[0, 0].bar(range(len(models)), mae_means, yerr=mae_stds, 
                   color=colors, alpha=0.7, capsize=5)
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('MAE (dB)', fontsize=12)
    axes[0, 0].set_title('Mean Absolute Error Comparison (dB domain)', fontsize=14)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # RMSE对比（dB）
    axes[0, 1].bar(range(len(models)), rmse_means, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_ylabel('RMSE (dB)', fontsize=12)
    axes[0, 1].set_title('Root Mean Squared Error Comparison (dB domain)', fontsize=14)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # SSIM对比
    axes[1, 0].bar(range(len(models)), ssim_means, color=colors, alpha=0.7)
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_ylabel('SSIM', fontsize=12)
    axes[1, 0].set_title('Structural Similarity Index Comparison\n(Valid Region Only)', fontsize=14)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # 综合排名
    # 归一化分数（MAE和RMSE越小越好，SSIM越大越好）
    mae_arr = np.array(mae_means)
    rmse_arr = np.array(rmse_means)
    ssim_arr = np.array(ssim_means)
    
    # 处理可能的nan
    mae_arr = np.nan_to_num(mae_arr, nan=np.nanmax(mae_arr))
    rmse_arr = np.nan_to_num(rmse_arr, nan=np.nanmax(rmse_arr))
    ssim_arr = np.nan_to_num(ssim_arr, nan=0)
    
    mae_range = max(mae_arr) - min(mae_arr)
    rmse_range = max(rmse_arr) - min(rmse_arr)
    
    mae_normalized = 1 - (mae_arr - min(mae_arr)) / (mae_range + 1e-8)
    rmse_normalized = 1 - (rmse_arr - min(rmse_arr)) / (rmse_range + 1e-8)
    ssim_normalized = ssim_arr
    
    # 综合得分：MAE(40%) + RMSE(30%) + SSIM(30%)
    composite_score = 0.4 * mae_normalized + 0.3 * rmse_normalized + 0.3 * ssim_normalized
    
    bars = axes[1, 1].bar(range(len(models)), composite_score, color=colors, alpha=0.7)
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Composite Score', fontsize=12)
    axes[1, 1].set_title('Overall Performance Score\n(40%×MAE + 30%×RMSE + 30%×SSIM)', fontsize=14)
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    # 标注最佳模型
    best_idx = np.argmax(composite_score)
    axes[1, 1].text(best_idx, composite_score[best_idx] + 0.02, '★ Best', 
                   ha='center', fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_comparison_dB.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 保存对比图: {save_path}")
    plt.close()


# ==================== 主评估函数 ====================
def main():
    """主评估流程"""
    config = EvalConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    print(f"⭐ 评估模式: dB域")
    print(f"⭐ SSIM计算: 只在有效区域（排除建筑物和无效点）")
    print(f"⭐ 归一化参数: NO_LABEL={config.NO_LABEL_VALUE} dB, MAX={config.MAX_DB_VALUE} dB\n")
    
    # 创建结果目录
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    all_results = {}
    
    # 遍历所有模型
    for model_config in config.MODELS:
        model_name = model_config['name']
        model_path = model_config['path']
        
        print("\n" + "="*80)
        print(f"评估模型: {model_name}")
        print("="*80)
        
        # 检查模型文件
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            continue
        
        try:
            # 创建测试数据集
            print(f"创建测试数据集...")
            feature_map_dir = model_config.get('feature_map_dir', config.DIR_FEATURE_MAPS)
            
            test_dataset = MultiBeamRadioDataset(
                phase="test",
                dir_multibeam=config.DIR_MULTIBEAM,
                dir_height_maps=config.DIR_HEIGHT_MAPS,
                dir_feature_maps=feature_map_dir,
                split_strategy=model_config['split_strategy'],
                train_ratio=config.TRAIN_RATIO,
                val_ratio=config.VAL_RATIO,
                test_ratio=config.TEST_RATIO,
                mode=model_config['task_mode'],
                random_seed=config.RANDOM_SEED,
                use_3d_buildings=True,
                use_feature_maps=model_config['use_feature_maps'],
                use_continuous_encoding=not model_config['use_feature_maps']
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=2
            )
            
            print(f"测试集样本数: {len(test_dataset)}")
            print(f"输入通道配置: {model_config['input_channels']}")
            
            # 加载模型
            print(f"加载模型...")
            if model_name=='Solution1_environment':
                model = RadioWNet(
                    inputs=test_dataset.input_channels-1, 
                    phase="secondU", 
                    use_film=False
                )
            else:
                model = RadioWNet(
                    inputs=test_dataset.input_channels, 
                    phase="secondU", 
                    use_film=False
                )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            # 评估
            print(f"开始评估...")
            summary, preds_norm, targets_norm, preds_dB, targets_dB, masks = evaluate_model(
                model, test_loader, device, model_config,
                config.NO_LABEL_VALUE, config.MAX_DB_VALUE
            )
            
            # 保存结果
            all_results[model_name] = summary
            
            # 打印结果（dB域）
            print(f"\n{model_name} 评估结果（dB域）:")
            print(f"  MAE:  {summary['mae_dB_mean']:.4f} ± {summary['mae_dB_std']:.4f} dB")
            print(f"  MSE:  {summary['mse_dB_mean']:.4f} ± {summary['mse_dB_std']:.4f} dB²")
            print(f"  RMSE: {summary['rmse_dB_mean']:.4f} ± {summary['rmse_dB_std']:.4f} dB")
            print(f"  SSIM: {summary['ssim_mean']:.6f} ± {summary['ssim_std']:.6f} (有效区域)")
            print(f"  测试样本数: {summary['num_samples']}")
            print(f"  有效SSIM样本数: {summary.get('num_valid_ssim_samples', summary['num_samples'])}")
            
            # 可视化
            if preds_norm:
                print(f"\n生成可视化...")
                visualize_predictions(
                    preds_norm, targets_norm, preds_dB, targets_dB, masks,
                    model_name, config.RESULTS_DIR
                )
            
        except Exception as e:
            print(f"❌ 评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建汇总
    if all_results:
        print("\n" + "="*80)
        print("生成汇总结果")
        print("="*80)
        
        df = create_summary_table(all_results, config.RESULTS_DIR)
        create_comparison_plots(all_results, config.RESULTS_DIR)
        
        print(f"\n🎉 评估完成！")
        print(f"结果保存在: {config.RESULTS_DIR}/")
        print(f"\n生成的文件:")
        print(f"  - evaluation_summary_dB.csv (dB域指标汇总)")
        print(f"  - metrics_comparison_dB.png (dB域对比图表)")
        print(f"  - {{model_name}}_visualization.png (各模型可视化)")
    else:
        print("\n❌ 没有成功评估的模型")


if __name__ == "__main__":
    main()