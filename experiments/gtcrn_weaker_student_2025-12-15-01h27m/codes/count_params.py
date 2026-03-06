"""
快速計算 GTCRN 模型參數量
"""
import torch
from omegaconf import OmegaConf
from models.gtcrn_end2end import GTCRN as Model

def count_parameters(model):
    """計算模型參數量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    # 讀取配置
    config_path = '/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_custom_2025-12-10-01h59m/config.yaml'
    cfg = OmegaConf.load(config_path)
    network_config = cfg['network_config']
    
    print("=" * 60)
    print("GTCRN 模型參數統計")
    print("=" * 60)
    print(f"\n模型配置:")
    for key, value in network_config.items():
        print(f"  {key}: {value}")
    
    # 創建模型
    model = Model(**network_config)
    
    # 計算參數量
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n參數統計:")
    print(f"  總參數量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可訓練參數: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  不可訓練參數: {total_params - trainable_params:,}")
    
    # 計算模型大小 (假設 float32)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    print(f"\n模型大小 (float32): {model_size_mb:.2f} MB")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
