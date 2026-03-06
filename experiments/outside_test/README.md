# Outside Test - GTCRN 音频降噪测试

本目录用于测试GTCRN模型的音频降噪效果，使用独立的测试脚本，不会影响原始实验数据。

## 使用的模型

实验路径: `/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_weaker_student_2025-12-15-01h27m`
模型检查点: `best_model_150.tar` (第150轮训练的最佳模型)

## 文件说明

- `noisy.wav` - 输入的带噪音频
- `enhanced.wav` - 模型处理后的增强音频
- `clean.wav` - (可选) 干净的参考音频，用于评估
- `test_single_audio.py` - 音频处理脚本
- `evaluate_audio.py` - 音频质量评估脚本
- `evaluation_results.txt` - 评估结果报告

## 使用方法

### 1. 处理音频

运行以下命令对 `noisy.wav` 进行降噪处理：

```bash
cd /home/sbplab/yuchen/GTCRN/SEtrain/experiments/outside_test
python test_single_audio.py
```

这将生成 `enhanced.wav` 文件。

### 2. 评估音频质量 (需要clean参考音频)

如果你有对应的干净音频作为参考，可以运行评估脚本：

```bash
python evaluate_audio.py
```

评估指标包括：
- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio) - 越高越好
- **SNR** (Signal-to-Noise Ratio) - 越高越好
- **PESQ** (Perceptual Evaluation of Speech Quality) - 1.0~4.5，越高越好
- **STOI** (Short-Time Objective Intelligibility) - 0~1，越高越好

评估结果会保存到 `evaluation_results.txt`。

### 3. 安装评估所需的库 (可选)

如果要使用PESQ和STOI评估：

```bash
pip install pesq pystoi
```

## 脚本参数说明

### test_single_audio.py

脚本内的主要参数：
- `checkpoint_path`: 模型检查点路径
- `input_audio`: 输入音频路径
- `output_audio`: 输出音频路径
- `device`: 'cuda' 或 'cpu'
- `chunk_size`: 处理片段长度（秒），默认2.0秒
- `sr`: 采样率，默认16000 Hz

### evaluate_audio.py

脚本内的主要参数：
- `clean_path`: 干净参考音频路径
- `noisy_path`: 带噪音频路径
- `enhanced_path`: 增强后音频路径
- `sr`: 采样率，默认16000 Hz

## 处理流程

1. **模型加载**: 从指定的checkpoint加载GTCRN模型
2. **音频读取**: 读取输入音频文件
3. **分段处理**: 
   - 对于短音频 (≤2秒): 直接处理
   - 对于长音频 (>2秒): 分段处理，使用50%重叠的滑动窗口
4. **重叠相加**: 使用汉宁窗进行平滑拼接
5. **正规化**: 确保输出在 [-1, 1] 范围内
6. **保存结果**: 输出增强后的音频

## 注意事项

1. 输入音频必须是16kHz采样率的单声道WAV文件
2. 模型在GPU上运行，如果没有GPU会自动切换到CPU（速度较慢）
3. 处理长音频时会自动分段，避免显存不足
4. 所有处理都在本目录进行，不会影响原始实验数据
5. 如果没有clean参考音频，可以跳过评估步骤，直接听音频进行主观评价

## 处理结果

运行 `test_single_audio.py` 后的输出示例：

```
使用设备: cuda
正在载入模型: .../best_model_150.tar
✅ 模型载入成功
   训练轮数: 150

正在处理音频: .../noisy.wav
   音频长度: 51520 采样点 (3.22 秒)
   采样率: 16000 Hz
   使用分段推理模式 (音频较长)
   将处理 2 个片段...
   正在拼接片段...
✅ 处理完成，已保存到: .../enhanced.wav

============================================================
测试完成!
============================================================
```

## 模型信息

- **架构**: GTCRN (Grouped Temporal Convolutional Recurrent Network)
- **参数量**: ~23.67K
- **计算量**: ~33.0 MMACs
- **训练数据**: DNS Challenge 数据集
- **采样率**: 16kHz
- **FFT参数**: n_fft=512, hop_length=256, win_length=512
