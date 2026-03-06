#!/usr/bin/env python3
"""
GTCRN 模型推論腳本
使用訓練好的模型對音訊進行降噪處理
"""
import os
import argparse
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.gtcrn_end2end import GTCRN


def load_model(checkpoint_path, device='cuda'):
    """
    載入訓練好的模型
    
    Args:
        checkpoint_path: checkpoint 文件路徑
        device: 運算設備 ('cuda' 或 'cpu')
    
    Returns:
        model: 載入的模型
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # 創建模型
    model = GTCRN(n_fft=512, hop_len=256, win_len=512)
    
    # 載入權重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 處理 DDP 訓練的 checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 移除 'module.' 前綴（如果有的話，來自 DDP 訓練）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    
    return model


def process_audio(model, audio_path, output_path, device='cuda', chunk_size=2.0, sr=16000):
    """
    處理單個音訊文件
    
    Args:
        model: GTCRN 模型
        audio_path: 輸入音訊路徑
        output_path: 輸出音訊路徑
        device: 運算設備
        chunk_size: 處理的片段長度（秒）
        sr: 採樣率
    """
    # 讀取音訊
    audio, fs = sf.read(audio_path, dtype='float32')
    
    if fs != sr:
        print(f"⚠️  Warning: Sample rate mismatch. Expected {sr}, got {fs}")
        return None
    
    # 轉為 tensor
    audio_tensor = torch.from_numpy(audio).to(device)
    
    # 如果音訊很短，直接處理
    chunk_length = int(chunk_size * sr)
    
    if len(audio_tensor) <= chunk_length:
        # 補零到 chunk_length
        if len(audio_tensor) < chunk_length:
            padding = chunk_length - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        
        with torch.no_grad():
            enhanced = model(audio_tensor.unsqueeze(0))[0]
        
        # 移除補零部分
        enhanced = enhanced[:len(audio)].cpu().numpy()
    
    else:
        # 長音訊：分段處理並拼接
        enhanced_chunks = []
        hop_length = chunk_length // 2  # 50% 重疊
        
        for start in range(0, len(audio_tensor), hop_length):
            end = min(start + chunk_length, len(audio_tensor))
            chunk = audio_tensor[start:end]
            
            # 補零到 chunk_length
            if len(chunk) < chunk_length:
                padding = chunk_length - len(chunk)
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            
            with torch.no_grad():
                enhanced_chunk = model(chunk.unsqueeze(0))[0]
            
            enhanced_chunks.append(enhanced_chunk.cpu().numpy())
            
            if end >= len(audio_tensor):
                break
        
        # 使用重疊相加法拼接
        enhanced = overlap_add(enhanced_chunks, hop_length, len(audio))
    
    # 正規化到 [-1, 1]
    max_val = np.abs(enhanced).max()
    if max_val > 1.0:
        enhanced = enhanced / max_val
    
    # 儲存
    sf.write(output_path, enhanced, sr)
    
    return enhanced


def overlap_add(chunks, hop_length, target_length):
    """
    重疊相加法拼接音訊片段
    
    Args:
        chunks: 音訊片段列表
        hop_length: 跳躍長度
        target_length: 目標長度
    
    Returns:
        拼接後的音訊
    """
    chunk_length = len(chunks[0])
    output = np.zeros(target_length)
    window = np.hanning(chunk_length)
    
    for i, chunk in enumerate(chunks):
        start = i * hop_length
        end = min(start + chunk_length, target_length)
        
        chunk_len = end - start
        output[start:end] += chunk[:chunk_len] * window[:chunk_len]
    
    return output


def batch_inference(model, input_dir, output_dir, device='cuda', file_ext='wav'):
    """
    批次推論：處理整個目錄的音訊
    
    Args:
        model: GTCRN 模型
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        device: 運算設備
        file_ext: 文件副檔名
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取所有音訊文件
    input_path = Path(input_dir)
    audio_files = list(input_path.glob(f'*.{file_ext}'))
    
    if len(audio_files) == 0:
        print(f"❌ No {file_ext} files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Processing...")
    
    # 處理每個文件
    for audio_file in tqdm(audio_files):
        output_file = os.path.join(output_dir, audio_file.name)
        
        try:
            process_audio(model, str(audio_file), output_file, device)
        except Exception as e:
            print(f"\n❌ Error processing {audio_file.name}: {e}")
            continue
    
    print(f"\n✅ Batch inference completed!")
    print(f"   Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='GTCRN 模型推論')
    
    # 模型參數
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路徑')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='運算設備 (default: cuda)')
    
    # 輸入輸出
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='輸入音訊文件或目錄')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='輸出音訊文件或目錄')
    
    # 處理參數
    parser.add_argument('--chunk-size', type=float, default=2.0,
                        help='處理的片段長度（秒）(default: 2.0)')
    parser.add_argument('--batch', action='store_true',
                        help='批次處理模式（處理整個目錄）')
    
    args = parser.parse_args()
    
    # 檢查設備
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("GTCRN 音訊降噪推論")
    print("=" * 60)
    print(f"Device: {args.device}")
    
    # 載入模型
    model = load_model(args.checkpoint, args.device)
    
    # 推論
    print("\n" + "=" * 60)
    if args.batch:
        # 批次處理
        batch_inference(model, args.input, args.output, args.device)
    else:
        # 單文件處理
        print(f"Processing: {args.input}")
        process_audio(model, args.input, args.output, args.device, args.chunk_size)
        print(f"✅ Enhanced audio saved to: {args.output}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
