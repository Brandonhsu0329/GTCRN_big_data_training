#!/usr/bin/env python3
"""
LDV 資料降噪測試腳本
"""
import sys
import torch
import soundfile as sf
import numpy as np
from scipy import signal
from pathlib import Path

# 添加專案路徑
PROJECT_ROOT = Path("/home/sbplab/yuchen/GTCRN/SEtrain")
sys.path.insert(0, str(PROJECT_ROOT))

from models.gtcrn_end2end import GTCRN


def load_model(checkpoint_path, device='cuda'):
    """載入訓練好的模型"""
    print(f"Loading model from: {checkpoint_path}")

    model = GTCRN(n_fft=512, hop_len=256, win_len=512)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")

    return model


def process_audio(model, audio_path, output_path, device='cuda', sr=16000):
    """處理單個音訊文件"""
    audio, fs = sf.read(audio_path, dtype='float32')
    original_sr = fs

    # Resample 到目標採樣率
    if fs != sr:
        print(f"  Resampling: {fs} Hz -> {sr} Hz")
        num_samples = int(len(audio) * sr / fs)
        audio = signal.resample(audio, num_samples)
        fs = sr

    audio_tensor = torch.from_numpy(audio).to(device)
    chunk_length = int(2.0 * sr)  # 2 秒 chunks

    if len(audio_tensor) <= chunk_length:
        if len(audio_tensor) < chunk_length:
            padding = chunk_length - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        with torch.no_grad():
            enhanced = model(audio_tensor.unsqueeze(0))[0]

        enhanced = enhanced[:len(audio)].cpu().numpy()

    else:
        # 長音訊分段處理
        enhanced_chunks = []
        hop_length = chunk_length // 2

        for start in range(0, len(audio_tensor), hop_length):
            end = min(start + chunk_length, len(audio_tensor))
            chunk = audio_tensor[start:end]

            if len(chunk) < chunk_length:
                padding = chunk_length - len(chunk)
                chunk = torch.nn.functional.pad(chunk, (0, padding))

            with torch.no_grad():
                enhanced_chunk = model(chunk.unsqueeze(0))[0]

            enhanced_chunks.append(enhanced_chunk.cpu().numpy())

            if end >= len(audio_tensor):
                break

        # 重疊相加 (使用 normalization 避免 fade-in/fade-out)
        output = np.zeros(len(audio))
        window_sum = np.zeros(len(audio))
        window = np.hanning(chunk_length)

        for i, chunk in enumerate(enhanced_chunks):
            start = i * hop_length
            end = min(start + chunk_length, len(audio))
            chunk_len = end - start
            output[start:end] += chunk[:chunk_len] * window[:chunk_len]
            window_sum[start:end] += window[:chunk_len]

        window_sum = np.maximum(window_sum, 1e-8)
        enhanced = output / window_sum

    # 正規化
    max_val = np.abs(enhanced).max()
    if max_val > 1.0:
        enhanced = enhanced / max_val

    # Resample 回原始採樣率
    if original_sr != sr:
        print(f"  Resampling back: {sr} Hz -> {original_sr} Hz")
        num_samples = int(len(enhanced) * original_sr / sr)
        enhanced = signal.resample(enhanced, num_samples)

    sf.write(output_path, enhanced, original_sr)
    return enhanced


def main():
    script_dir = Path(__file__).parent
    checkpoint_path = Path("/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_weaker_student_2025-12-15-01h27m/checkpoints/best_model_150.tar")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = load_model(str(checkpoint_path), device)

    # 處理所有 *_noisy.wav 檔案
    noisy_files = list(script_dir.glob("*_noisy.wav"))

    print(f"\nFound {len(noisy_files)} noisy files to process")
    print("=" * 60)

    for noisy_path in noisy_files:
        # 輸出檔名: sample00_noisy.wav -> sample00_enhanced.wav
        enhanced_name = noisy_path.name.replace("_noisy.wav", "_enhanced.wav")
        enhanced_path = script_dir / enhanced_name

        print(f"Processing {noisy_path.name}...")

        try:
            process_audio(model, str(noisy_path), str(enhanced_path), device)
            print(f"  -> Saved: {enhanced_name}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("=" * 60)
    print("Denoising completed!")


if __name__ == "__main__":
    main()
