# GTCRN 推論使用指南

## 📋 推論腳本功能

已創建兩個推論腳本：

1. **`inference.py`** - 主要推論腳本（Python）
2. **`run_inference.sh`** - 互動式推論腳本（Bash）

---

## 🚀 快速開始

### 方法 1: 互動式推論（最簡單）

```bash
cd /home/sbplab/yuchen/GTCRN/SEtrain
bash run_inference.sh
```

會出現選單：
```
1) 單個文件推論
2) 批次推論（處理整個目錄）
3) 測試驗證集樣本
```

### 方法 2: 命令行推論

#### 單個文件
```bash
python inference.py \
    -c experiments/gtcrn_custom_2025-12-10-00h28m/checkpoints/best_model_001.tar \
    -i /path/to/noisy.wav \
    -o /path/to/enhanced.wav \
    -d cuda
```

#### 批次處理整個目錄
```bash
python inference.py \
    -c experiments/gtcrn_custom_2025-12-10-00h28m/checkpoints/best_model_001.tar \
    -i /path/to/noisy_dir/ \
    -o /path/to/enhanced_dir/ \
    -d cuda \
    --batch
```

---

## 📖 參數說明

### 必需參數

- `-c, --checkpoint`: 模型 checkpoint 路徑
  - 例如: `experiments/.../best_model_001.tar`
  
- `-i, --input`: 輸入音訊文件或目錄
  - 單文件: `/path/to/noisy.wav`
  - 目錄: `/path/to/noisy_dir/`
  
- `-o, --output`: 輸出音訊文件或目錄
  - 單文件: `/path/to/enhanced.wav`
  - 目錄: `/path/to/enhanced_dir/`

### 可選參數

- `-d, --device`: 運算設備
  - `cuda` (默認，使用 GPU)
  - `cpu` (使用 CPU，較慢)
  
- `--chunk-size`: 處理的片段長度（秒）
  - 默認: 2.0 秒
  - 建議: 與訓練時的長度一致
  
- `--batch`: 啟用批次處理模式
  - 用於處理整個目錄

---

## 💡 使用範例

### 範例 1: 測試驗證集

處理所有驗證集樣本並保存結果：

```bash
python inference.py \
    -c experiments/gtcrn_custom_2025-12-10-00h28m/checkpoints/best_model_001.tar \
    -i datasets/val_data/train_noisy \
    -o inference_results/val_enhanced \
    -d cuda \
    --batch
```

然後你可以比較：
- **原始含噪**: `datasets/val_data/train_noisy/`
- **降噪後**: `inference_results/val_enhanced/`
- **真實乾淨**: `datasets/val_data/train_clean/`

### 範例 2: 處理自己的音訊

```bash
# 單個文件
python inference.py \
    -c experiments/gtcrn_custom_2025-12-10-00h28m/checkpoints/best_model_001.tar \
    -i /home/user/my_noisy_audio.wav \
    -o /home/user/my_enhanced_audio.wav
```

### 範例 3: 批次處理多個文件

```bash
# 準備輸入目錄
mkdir -p test_audio/noisy
# 複製你的含噪音訊到 test_audio/noisy/

# 批次推論
python inference.py \
    -c experiments/gtcrn_custom_2025-12-10-00h28m/checkpoints/best_model_001.tar \
    -i test_audio/noisy \
    -o test_audio/enhanced \
    --batch
```

### 範例 4: CPU 推論（無 GPU）

```bash
python inference.py \
    -c experiments/gtcrn_custom_2025-12-10-00h28m/checkpoints/best_model_001.tar \
    -i input.wav \
    -o output.wav \
    -d cpu
```

---

## 📊 Checkpoint 選擇

你有 3 個 checkpoint 可選：

```
checkpoints/
├── best_model_001.tar    # 最佳模型（推薦）✅
├── model_001.tar         # Epoch 1
└── model_002.tar         # Epoch 2
```

**建議使用**: `best_model_001.tar`（根據驗證 PESQ 選出的最佳模型）

---

## 🔧 進階功能

### 1. 處理長音訊

腳本會自動處理長音訊：
- 分段處理（默認 2 秒片段）
- 50% 重疊
- 使用 overlap-add 拼接

```bash
# 處理 10 秒的音訊
python inference.py \
    -c checkpoints/best_model_001.tar \
    -i long_audio.wav \
    -o long_enhanced.wav \
    --chunk-size 2.0
```

### 2. 處理不同採樣率

⚠️ **注意**: 模型訓練在 16 kHz，輸入音訊必須是 16 kHz

如果你的音訊不是 16 kHz，需要先轉換：

```bash
# 使用 sox
sox input_44100.wav -r 16000 input_16000.wav

# 使用 ffmpeg
ffmpeg -i input_44100.wav -ar 16000 input_16000.wav
```

### 3. Python API 使用

也可以在 Python 代碼中直接使用：

```python
from inference import load_model, process_audio

# 載入模型
model = load_model('checkpoints/best_model_001.tar', device='cuda')

# 處理音訊
process_audio(
    model, 
    'input.wav', 
    'output.wav', 
    device='cuda',
    chunk_size=2.0
)
```

---

## 🎯 性能提示

### GPU vs CPU 速度對比

| 設備 | 處理 2 秒音訊 | 處理 10 秒音訊 |
|------|-------------|---------------|
| RTX 2080 Ti | ~0.1 秒 | ~0.5 秒 |
| CPU (估計) | ~2 秒 | ~10 秒 |

### 優化建議

1. **使用 GPU**: 速度提升 10-20 倍
2. **批次處理**: 使用 `--batch` 處理多個文件更高效
3. **適當的 chunk_size**: 
   - 太小: 增加處理次數
   - 太大: 可能 OOM
   - 建議: 2-4 秒

---

## ❗ 常見問題

### Q1: CUDA out of memory
**解決**: 減小 `--chunk-size`
```bash
python inference.py ... --chunk-size 1.0
```

### Q2: 採樣率不匹配
**錯誤**: `Sample rate mismatch. Expected 16000, got 44100`

**解決**: 先轉換採樣率
```bash
sox input.wav -r 16000 input_16k.wav
```

### Q3: 效果不好
**原因**: 
- 只訓練了 2 epochs（需要更多訓練）
- 數據集較小
- 噪音類型與訓練數據不同

**建議**:
- 繼續訓練到 50-100 epochs
- 使用完整訓練的模型
- 在相似噪音環境下使用

### Q4: 處理速度慢
**解決**:
1. 使用 GPU（`-d cuda`）
2. 確保 CUDA 正確安裝
3. 批次處理多個文件

---

## 📝 輸出文件格式

- **格式**: WAV
- **採樣率**: 16000 Hz
- **位元深度**: 32-bit float
- **聲道**: 單聲道
- **正規化**: [-1, 1]

---

## 🔍 驗證推論結果

### 方法 1: 聽覺比較
使用音訊播放器比較：
- 原始含噪音訊
- 降噪後音訊
- 真實乾淨音訊（如有）

### 方法 2: 計算客觀指標

可以使用評估腳本計算 PESQ, STOI 等指標：

```bash
python evaluate.py \
    --clean_dir datasets/val_data/train_clean \
    --enhanced_dir inference_results/val_enhanced
```

---

## 📦 完整工作流程

```bash
# 1. 準備輸入音訊（確保 16kHz）
mkdir -p test_input
cp your_noisy_audio.wav test_input/

# 2. 運行推論
python inference.py \
    -c experiments/gtcrn_custom_2025-12-10-00h28m/checkpoints/best_model_001.tar \
    -i test_input \
    -o test_output \
    --batch

# 3. 查看結果
ls test_output/

# 4. 播放比較
# 原始: test_input/your_noisy_audio.wav
# 降噪: test_output/your_noisy_audio.wav
```

---

## ✅ 檢查清單

推論前確認：

- [ ] Checkpoint 文件存在
- [ ] 輸入音訊是 16 kHz
- [ ] 輸入音訊是 WAV 格式
- [ ] GPU 可用（如使用 CUDA）
- [ ] 輸出目錄可寫入

---

**準備好了！現在你可以使用訓練好的模型進行音訊降噪了！** 🎉

如有問題，請查看錯誤訊息或使用 `python inference.py --help` 查看完整選項。
