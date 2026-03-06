# 🧪 弱化學生模型訓練實驗

## 📋 實驗基本信息

**實驗名稱**: `gtcrn_weaker_student_2025-12-15-01h30m`  
**創建日期**: 2025-12-15  
**實驗目標**: 降低學生模型性能,創造更大的師生差距以改善知識蒸餾效果

## 🎯 實驗目標

### 性能目標
- **原始學生 PESQ**: 2.3909 (驗證集) / 2.5043 (測試集)
- **目標 PESQ**: 2.15-2.25 (驗證集) / 2.30-2.40 (測試集, 預估)
- **性能降低幅度**: -0.14 ~ -0.24 PESQ

### 知識蒸餾目標
- **原始師生差距**: 0.1043 (測試集: 2.6086 vs 2.5043)
- **目標師生差距**: 0.30-0.40 (測試集: 2.60-2.70 vs 2.30-2.40)
- **改善原因**: 更大的性能差距有助於知識蒸餾的梯度傳遞

## 📊 基準實驗

**基準實驗**: `gtcrn_custom_2025-12-10-01h59m_1st`

### 基準結果
- 訓練 Epochs: 256
- 最佳驗證 PESQ: 2.3909 @ Epoch 243
- 最終驗證 PESQ: 2.3456 @ Epoch 256
- 測試集 PESQ: 2.5043
- 初始學習率: 0.001
- 最終學習率: 0.000808 (衰減 19.2%)
- 過擬合程度: Val-Train Gap = 0.2513

### 基準配置
```yaml
optimizer:
  lr: 0.001
  weight_decay: 0  # 無正則化

scheduler:
  kwargs:
    warmup_steps: 2000
    decay_until_step: 50000
    max_lr: 0.001
    min_lr: 1.0e-06

trainer:
  epochs: 256
  clip_grad_norm_value: 3.0
```

## 🔧 本次實驗配置

### 策略: 組合策略 (Strategy G)
結合三種機制降低模型性能:
1. **提早停止訓練** - 限制學習時間
2. **降低學習率** - 限制學習速度
3. **增加正則化** - 限制模型複雜度

### 修改的超參數

| 參數 | 原始值 | 新值 | 修改原因 | 預期影響 |
|------|--------|------|---------|---------|
| `optimizer.lr` | 0.001 | **0.0005** | 降低學習速度 | PESQ -0.09~-0.14 |
| `optimizer.weight_decay` | 0 (無) | **1.0e-04** | 增加 L2 正則化 | PESQ -0.06~-0.09 |
| `scheduler.kwargs.max_lr` | 0.001 | **0.0005** | 匹配 optimizer.lr | 一致性 |
| `trainer.epochs` | 256 | **150** | 提早停止訓練 | PESQ -0.13 |

### 完整配置
```yaml
optimizer:
  lr: 0.0005              # ⬇️ 降低至原來的 50%
  weight_decay: 1.0e-04   # ➕ 新增較強正則化

scheduler:
  kwargs:
    warmup_steps: 2000
    decay_until_step: 50000
    max_lr: 0.0005        # ⬇️ 匹配 optimizer.lr
    min_lr: 1.0e-06

trainer:
  epochs: 150             # ⬇️ 提早至 Epoch 150 停止
  clip_grad_norm_value: 3.0
  exp_path: experiments/gtcrn_weaker_student
```

### 保持不變的參數
- 模型架構: GTCRN (16 channels, 1 DPGRNN layer)
- Batch size: 16
- 訓練數據量: num_data_per_epoch = 1000
- 梯度裁剪: clip_grad_norm_value = 3.0
- Loss 配置: lamda_ri=30, lamda_mag=70
- 數據增強: 無

## 📈 預期結果分析

### 各策略獨立效果 (參考)
- **策略 A (提早停止至 150)**: PESQ -0.13
- **策略 B (降低 LR 至 0.0005)**: PESQ -0.09~-0.14
- **策略 D (weight_decay 1e-4)**: PESQ -0.06~-0.09

### 組合效果預估
**累積降幅**: -0.14 ~ -0.24 PESQ  
**目標驗證 PESQ**: 2.15 - 2.25  
**目標測試 PESQ**: 2.30 - 2.40 (假設 test > val + 0.15)

### 訓練過程預期
1. **Epoch 1-50**: 快速學習階段
   - 學習率從 0 warmup 至 0.0005
   - PESQ 快速提升至 ~1.90-2.00
   
2. **Epoch 51-100**: 穩定提升階段
   - 學習率緩慢衰減
   - PESQ 提升至 ~2.10-2.18
   - Weight decay 開始抑制過擬合

3. **Epoch 101-150**: 收尾階段
   - 學習率進一步衰減
   - PESQ 達到最終 2.15-2.25
   - 提早停止,避免進一步優化

## ✅ 成功標準

### 主要指標
- ✅ 驗證 PESQ: 2.15 - 2.25 (降低 0.14-0.24)
- ✅ 測試 PESQ: 2.30 - 2.40 (預估)
- ✅ 師生差距: 0.30 - 0.40 (vs 老師 2.60-2.70)

### 次要指標
- 訓練穩定: 無嚴重震蕩
- 過擬合控制: Val-Train Gap < 0.35
- 收斂性: 最後 20 epochs PESQ std < 0.05

## 🔬 對照實驗矩陣

| 實驗名稱 | LR | Weight Decay | Epochs | 預期 PESQ | 用途 |
|---------|-----|--------------|--------|-----------|------|
| gtcrn_custom_2025-12-10 | 0.001 | 0 | 256 | 2.39 | 基準 (原始) |
| **gtcrn_weaker_2025-12-15** | **0.0005** | **1e-4** | **150** | **2.15-2.25** | **弱化學生** |
| teacher_large_optimized | 0.001 | 5e-6 | 256 | 2.55-2.62 | 優化老師 |

## 📝 實驗記錄

### 訓練命令
```bash
cd /home/sbplab/yuchen/GTCRN/SEtrain
python train.py -C experiments/gtcrn_weaker_student_2025-12-15-01h30m/config.yaml
```

### 評估命令
```bash
# 驗證集評估 (自動)
# 在訓練過程中每個 epoch 自動執行

# 測試集評估
cd /home/sbplab/yuchen/GTCRN/SEtrain
python evaluate.py \
    -C experiments/gtcrn_weaker_student_2025-12-15-01h30m/config.yaml \
    -M experiments/gtcrn_weaker_student_2025-12-15-01h30m/checkpoints/best_model.pth \
    --test_dir /path/to/test/dataset
```

## 🎓 知識蒸餾下一步

若本實驗成功達到目標 (PESQ 2.15-2.25):

1. **訓練新老師模型**
   - 使用 teacher_config_optimized.yaml
   - 目標 PESQ: 2.60-2.70

2. **執行知識蒸餾**
   - Teacher: teacher_large_optimized (PESQ 2.60-2.70)
   - Student: gtcrn_weaker (PESQ 2.15-2.25)
   - 性能差距: 0.35-0.55 (理想範圍)

3. **預期蒸餾效果**
   - 蒸餾後學生 PESQ: 2.35-2.45 (+0.10-0.20)
   - 相比原始學生: -0.05~-0.15 (仍輕量但性能接近)

## 📌 注意事項

1. **訓練監控**
   - 監控 loss 曲線,確保穩定下降
   - 檢查過擬合程度 (Val-Train Gap)
   - 記錄最佳 checkpoint epoch

2. **結果驗證**
   - 若 PESQ > 2.25,考慮進一步降低 LR 或減少 epochs
   - 若 PESQ < 2.15,可能過度限制,考慮放寬參數

3. **對比分析**
   - 訓練完成後對比三個實驗的學習曲線
   - 分析各策略的實際貢獻度
   - 記錄實際師生差距

## 🗂️ 實驗文件結構

```
gtcrn_weaker_student_2025-12-15-01h30m/
├── config.yaml                    # 修改後的配置文件
├── EXPERIMENT_PLAN.md            # 本文件 - 實驗計劃書
├── checkpoints/                  # 訓練權重 (待生成)
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── codes/                        # 訓練代碼備份 (待生成)
├── evaluation_results/           # 評估結果 (待生成)
├── logs/                         # 訓練日誌 (待生成)
├── val_samples/                  # 驗證樣本 (待生成)
└── training_history.csv          # 訓練歷史 (待生成)
```

---

**實驗設計**: GitHub Copilot  
**審核狀態**: ⏳ 待開始  
**預計訓練時間**: ~3-4 小時 (150 epochs)
