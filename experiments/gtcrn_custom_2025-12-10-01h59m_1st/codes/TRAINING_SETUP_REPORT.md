# GTCRN 訓練準備完成報告

## ✅ 已完成的任務

### 1. 噪音重採樣 ✓
- **位置**: `SEtrain/speech_lib/noisex92_16k/`
- **內容**: 8 種噪音，已從 19.98 kHz 重採樣到 16 kHz
- **文件**: babble.wav, f16.wav, factory1.wav, factory2.wav, leopard.wav, pink.wav, volvo.wav, white.wav

### 2. RIR 生成 ✓
- **位置**: `SEtrain/speech_lib/rirs/`
- **內容**: 200 個房間脈衝響應
- **方法**: 簡單指數衰減模擬（如需更真實的 RIR，可安裝 pyroomacoustics）

### 3. 訓練數據生成 ✓
- **位置**: `SEtrain/datasets/`
- **訓練集**: `train_data/` - 1728 個樣本（90%）
- **驗證集**: `val_data/` - 192 個樣本（10%）
- **音訊長度**: 2 秒（32000 樣本 @ 16kHz）
- **處理方式**: 
  - 長度不足：自動補零
  - 長度超過：隨機裁切或從頭裁切

### 4. 數據配置 ✓
- **CSV 列表**:
  - `prepare_datasets/train_clean_dir.csv` - 1920 條乾淨語音路徑
  - `prepare_datasets/train_noise_dir.csv` - 噪音路徑（重複使用）
  - `prepare_datasets/train_rir_dir.csv` - RIR 路徑（重複使用）

### 5. Dataloader 更新 ✓
- **文件**: `dataloader_custom.py`
- **特點**:
  - 支持固定 2 秒長度
  - 自動補零或裁切
  - 讀取新的數據路徑
  
### 6. 訓練配置 ✓
- **文件**: `configs/cfg_train_custom.yaml`
- **配置**:
  - 音訊長度: 2 秒
  - Batch size: 16（因為音訊變短，可增大）
  - Epochs: 100
  - 學習率: 0.001 with warmup
  - 實驗輸出: `experiments/gtcrn_custom/`

---

## 📁 目錄結構

```
SEtrain/
├── speech_lib/              # 原始音訊數據
│   ├── boy1/ boy2/ boy3/   # 男生語音（各 320 句）
│   ├── girl1/ girl2/ girl3/# 女生語音（各 320 句）
│   ├── noisex92/           # 原始噪音（19.98 kHz）
│   ├── noisex92_16k/       # 重採樣後的噪音（16 kHz）✓
│   └── rirs/               # 生成的 RIR（200 個）✓
│
├── datasets/                # 生成的訓練數據 ✓
│   ├── train_data/
│   │   ├── train_noisy/    # 1728 個含噪語音（2秒）
│   │   ├── train_clean/    # 1728 個乾淨語音（2秒）
│   │   └── train_INFO.csv  # 數據生成記錄
│   └── val_data/
│       ├── train_noisy/    # 192 個驗證含噪語音
│       └── train_clean/    # 192 個驗證乾淨語音
│
├── prepare_datasets/        # 數據準備腳本
│   ├── resample_noise.py          # 噪音重採樣 ✓
│   ├── generate_rir.py            # RIR 生成 ✓
│   ├── generate_csv_lists.py      # CSV 列表生成 ✓
│   ├── gen_train_data.py          # 訓練數據生成 ✓
│   ├── train_clean_dir.csv        # 乾淨語音列表 ✓
│   ├── train_noise_dir.csv        # 噪音列表 ✓
│   └── train_rir_dir.csv          # RIR 列表 ✓
│
├── configs/
│   └── cfg_train_custom.yaml      # 自定義訓練配置 ✓
│
├── dataloader_custom.py           # 自定義 dataloader ✓
├── train.py                       # 訓練腳本
└── models/
    └── gtcrn_end2end.py           # GTCRN 模型
```

---

## 🚀 如何開始訓練

### 方法 1: 使用自定義配置（推薦）

```bash
cd /home/sbplab/yuchen/GTCRN/SEtrain

# 單卡訓練
python train.py -C configs/cfg_train_custom.yaml -D 0
```

**注意**: 需要修改 `train.py` 的第 21 行，將導入改為：
```python
from dataloader_custom import CustomDataset as Dataset
```

### 方法 2: 快速測試

```bash
# 先測試 dataloader
python dataloader_custom.py

# 測試一個 epoch
python train.py -C configs/cfg_train_custom.yaml -D 0
```

---

## 📊 數據統計

| 項目 | 數量 | 詳情 |
|------|------|------|
| **乾淨語音** | 1920 句 | 6 人 × 320 句，平均 3 秒/句 |
| **噪音類型** | 8 種 | NOISEX-92，每種約 4 分鐘 |
| **RIR** | 200 個 | 簡單指數衰減模擬 |
| **訓練樣本** | 1728 個 | 2 秒/樣本，含噪+混響 |
| **驗證樣本** | 192 個 | 2 秒/樣本 |
| **SNR 範圍** | -5 ~ 15 dB | 隨機分佈 |
| **總時長** | ~64 分鐘 | 1920 × 2 秒 |

---

## ⚙️ 訓練參數

```yaml
音訊參數:
- 採樣率: 16 kHz
- 長度: 2 秒（32000 樣本）
- STFT: n_fft=512, hop=256, win=512

訓練參數:
- Batch size: 16
- Epochs: 100
- 優化器: Adam (lr=0.001)
- 學習率調度: Warmup + Cosine Annealing
- 梯度裁剪: 3.0

損失函數:
- 30 × (Real + Imag Loss)
- 70 × Magnitude Loss
- 1 × SI-SNR Loss

每個 Epoch:
- 訓練步數: ~62 batches
- 驗證步數: ~12 batches
- 預計時間: ~5-10 分鐘（取決於 GPU）
```

---

## 🔍 需要注意的問題

### ⚠️ 已知限制

1. **數據量偏小**
   - 只有 1.58 小時的乾淨語音
   - 可能導致過擬合
   - 建議：監控訓練/驗證 loss 差距

2. **說話人數量少**
   - 只有 6 個人
   - 對未見過的說話人泛化能力有限
   - 建議：如需更好泛化，補充更多說話人數據

3. **RIR 簡化**
   - 使用簡單指數衰減，不是真實房間響應
   - 建議：安裝 pyroomacoustics 生成更真實的 RIR
   ```bash
   pip install pyroomacoustics
   python prepare_datasets/generate_rir.py  # 重新生成
   ```

4. **音訊長度較短**
   - 2 秒可能不足以學習長時序依賴
   - 如有需要可改回 4-6 秒
   - 修改 `cfg_train_custom.yaml` 中的 `length_in_seconds`

---

## 🛠️ 可選改進

### 1. 增加數據多樣性
```bash
# 下載 LibriSpeech 補充數據（5 小時）
cd speech_lib
wget http://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz

# 重新生成 CSV 和訓練數據
cd ../prepare_datasets
python generate_csv_lists.py
python gen_train_data.py
```

### 2. 數據增強
在訓練時可加入：
- SpecAugment（頻譜遮罩）
- 語速擾動（Speed Perturbation）
- 多種 SNR 混合

### 3. 監控訓練
```bash
# 啟動 TensorBoard
tensorboard --logdir experiments/gtcrn_custom/
```

---

## ✅ 檢查清單

在開始訓練前，請確認：

- [x] 噪音已重採樣到 16 kHz
- [x] RIR 已生成（200 個）
- [x] 訓練數據已生成（1728 個樣本）
- [x] 驗證數據已分離（192 個樣本）
- [x] CSV 列表已創建
- [x] 配置文件已更新
- [x] Dataloader 已測試通過
- [ ] train.py 已修改導入（需要手動修改）
- [ ] GPU 可用（檢查 `nvidia-smi`）
- [ ] 磁碟空間充足（至少 10 GB）

---

## 🎯 預期效果

根據數據量評估：

| 指標 | 預期值 | 備註 |
|------|--------|------|
| **訓練時間** | 8-12 小時 | 100 epochs，單 GPU |
| **PESQ** | 2.3-2.5 | 較小數據集，低於論文的 2.87 |
| **過擬合風險** | 中等 | 需監控 train/val loss |
| **泛化能力** | 有限 | 對已知說話人效果好 |
| **實用性** | 可用 | 適合快速原型和測試 |

**建議**：
- 先訓練 20-30 epochs 看趨勢
- 如果驗證 loss 不再下降，提早停止
- 可嘗試不同的學習率和 batch size

---

## 📝 下一步

1. **修改 train.py 導入**（必需）
2. **測試訓練流程**（跑 1-2 epochs）
3. **監控訓練曲線**
4. **評估模型效果**
5. **根據需要調整參數或補充數據**

---

## 📞 問題排查

如遇到問題：

1. **OOM (Out of Memory)**
   - 減小 batch_size（16 → 8）
   - 縮短音訊長度（2s → 1s）

2. **Loss 不下降**
   - 檢查學習率（可能太大或太小）
   - 檢查數據是否正確加載
   - 嘗試降低 warmup_steps

3. **過擬合嚴重**
   - 提早停止訓練
   - 增加數據增強
   - 考慮補充更多數據

---

生成時間: 2024-12-09
狀態: ✅ 全部完成，準備開始訓練
