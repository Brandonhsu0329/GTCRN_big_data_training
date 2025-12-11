"""
檢查 checkpoint 內容
"""
import torch

ckpt_path = '/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_custom_2025-12-10-01h59m/checkpoints/best_model_250.tar'

# 載入 checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu')

print("=" * 60)
print("Checkpoint 內容分析")
print("=" * 60)

print("\n1. Checkpoint 包含的 keys:")
for key in ckpt.keys():
    print(f"   - {key}")

print("\n2. 各部分大小:")
total_size = 0
for key, value in ckpt.items():
    if isinstance(value, dict):
        # 計算 state_dict 大小
        size = sum(p.numel() * p.element_size() for p in value.values() if torch.is_tensor(p))
        size_mb = size / (1024 * 1024)
        print(f"   {key}: {size_mb:.2f} MB ({sum(p.numel() for p in value.values() if torch.is_tensor(p)):,} 參數)")
        total_size += size
    elif torch.is_tensor(value):
        size = value.numel() * value.element_size()
        size_mb = size / (1024 * 1024)
        print(f"   {key}: {size_mb:.2f} MB")
        total_size += size
    else:
        print(f"   {key}: {value} (非張量)")

print(f"\n3. 總大小: {total_size / (1024 * 1024):.2f} MB")

# 模型參數統計
if 'model' in ckpt:
    print("\n4. 模型參數詳細:")
    model_params = sum(p.numel() for p in ckpt['model'].values())
    print(f"   總參數量: {model_params:,} ({model_params/1e6:.2f}M)")

# 優化器狀態
if 'optimizer' in ckpt:
    print("\n5. 優化器狀態:")
    opt_state = ckpt['optimizer']['state']
    print(f"   狀態數量: {len(opt_state)}")
    if opt_state:
        first_key = list(opt_state.keys())[0]
        print(f"   每個狀態包含: {list(opt_state[first_key].keys())}")

print("\n" + "=" * 60)
print("結論:")
print("=" * 60)
print("模型文件較大的原因:")
print("  1. 包含模型參數 (model)")
print("  2. 包含優化器狀態 (optimizer) - Adam 會存儲動量和方差")
print("  3. 包含 scheduler 狀態")
print("  4. 包含 epoch 等訓練信息")
print("\n如果只需要模型權重，可以:")
print("  torch.save(ckpt['model'], 'model_only.pth')")
print("  這樣大小會接近 0.18 MB")
