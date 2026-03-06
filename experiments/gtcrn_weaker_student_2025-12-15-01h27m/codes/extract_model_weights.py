"""
提取純模型權重（不含優化器狀態）
"""
import torch
from pathlib import Path

# 來源 checkpoint
src_path = '/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_custom_2025-12-10-01h59m/checkpoints/best_model_250.tar'
# 輸出路徑（只含模型權重）
dst_path = '/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_custom_2025-12-10-01h59m/checkpoints/best_model_250_weights_only.pth'

# 載入完整 checkpoint
ckpt = torch.load(src_path, map_location='cpu')

# 只保存模型權重
torch.save(ckpt['model'], dst_path)

# 檢查大小
import os
orig_size = os.path.getsize(src_path) / 1024  # KB
new_size = os.path.getsize(dst_path) / 1024   # KB

print("=" * 60)
print("模型權重提取完成")
print("=" * 60)
print(f"原始文件 (.tar): {orig_size:.1f} KB")
print(f"純權重文件 (.pth): {new_size:.1f} KB")
print(f"減少: {orig_size - new_size:.1f} KB ({(1 - new_size/orig_size)*100:.1f}%)")
print(f"\n保存位置: {dst_path}")
print("\n使用方式:")
print("  model.load_state_dict(torch.load('best_model_250_weights_only.pth'))")
