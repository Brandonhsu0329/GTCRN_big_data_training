#!/usr/bin/env python3
"""
独立音频测试脚本
使用指定的GTCRN模型处理单个音频文件
"""
import os
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.gtcrn_end2end import GTCRN


def load_model(checkpoint_path, device='cuda'):
    """
    载入训练好的模型
    
    Args:
        checkpoint_path: checkpoint 文件路径
        device: 运算设备 ('cuda' 或 'cpu')
    
    Returns:
        model: 载入的模型
    """
    print(f"正在载入模型: {checkpoint_path}")
    
    # 创建模型
    model = GTCRN(n_fft=512, hop_len=256, win_len=512)
    
    # 载入权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理 DDP 训练的 checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 移除 'module.' 前缀（如果有的话，来自 DDP 训练）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型载入成功")
    if 'epoch' in checkpoint:
        print(f"   训练轮数: {checkpoint['epoch']}")
    if 'best_loss' in checkpoint:
        print(f"   最佳损失: {checkpoint['best_loss']:.6f}")
    
    return model


def process_audio(model, audio_path, output_path, device='cuda', chunk_size=2.0, sr=16000):
    """
    处理单个音频文件
    
    Args:
        model: GTCRN 模型
        audio_path: 输入音频路径
        output_path: 输出音频路径
        device: 运算设备
        chunk_size: 处理的片段长度（秒）
        sr: 采样率
    """
    print(f"\n正在处理音频: {audio_path}")
    
    # 读取音频
    audio, fs = sf.read(audio_path, dtype='float32')
    
    if fs != sr:
        print(f"⚠️  警告: 采样率不匹配. 期望 {sr} Hz, 但得到 {fs} Hz")
        print("   音频可能需要重采样")
    
    original_length = len(audio)
    print(f"   音频长度: {len(audio)} 采样点 ({len(audio)/fs:.2f} 秒)")
    print(f"   采样率: {fs} Hz")
    
    # 转为 tensor
    audio_tensor = torch.from_numpy(audio).to(device)
    
    # 如果音频很短，直接处理
    chunk_length = int(chunk_size * sr)
    
    if len(audio_tensor) <= chunk_length:
        # 补零到 chunk_length
        if len(audio_tensor) < chunk_length:
            padding = chunk_length - len(audio_tensor)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
        
        print(f"   使用单次推理模式 (音频较短)")
        with torch.no_grad():
            enhanced = model(audio_tensor.unsqueeze(0))[0]
        
        # 移除补零部分
        enhanced = enhanced[:original_length].cpu().numpy()
    
    else:
        # 长音频：分段处理并拼接
        print(f"   使用分段推理模式 (音频较长)")
        enhanced_chunks = []
        hop_length = chunk_length // 2  # 50% 重叠
        
        num_chunks = (len(audio_tensor) - chunk_length) // hop_length + 1
        print(f"   将处理 {num_chunks} 个片段...")
        
        for i, start in enumerate(range(0, len(audio_tensor), hop_length)):
            end = min(start + chunk_length, len(audio_tensor))
            chunk = audio_tensor[start:end]
            
            # 补零到 chunk_length
            if len(chunk) < chunk_length:
                padding = chunk_length - len(chunk)
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            
            with torch.no_grad():
                enhanced_chunk = model(chunk.unsqueeze(0))[0]
            
            enhanced_chunks.append(enhanced_chunk.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"   进度: {i+1}/{num_chunks} 片段")
            
            if end >= len(audio_tensor):
                break
        
        # 使用重叠相加法拼接
        print(f"   正在拼接片段...")
        enhanced = overlap_add(enhanced_chunks, hop_length, original_length)
    
    # 正规化到 [-1, 1]
    max_val = np.abs(enhanced).max()
    if max_val > 1.0:
        enhanced = enhanced / max_val
        print(f"   音频已正规化 (原始最大值: {max_val:.4f})")
    
    # 儲存
    sf.write(output_path, enhanced, sr)
    print(f"✅ 处理完成，已保存到: {output_path}")
    
    return enhanced


def overlap_add(chunks, hop_length, target_length):
    """
    重叠相加法拼接音频片段
    
    Args:
        chunks: 音频片段列表
        hop_length: 跳跃长度
        target_length: 目标长度
    
    Returns:
        拼接后的音频
    """
    chunk_length = len(chunks[0])
    output = np.zeros(target_length, dtype=np.float32)
    norm = np.zeros(target_length, dtype=np.float32)
    
    # 使用汉宁窗进行平滑
    window = np.hanning(chunk_length).astype(np.float32)
    
    for i, chunk in enumerate(chunks):
        start = i * hop_length
        end = min(start + chunk_length, target_length)
        chunk_end = end - start
        
        output[start:end] += chunk[:chunk_end] * window[:chunk_end]
        norm[start:end] += window[:chunk_end]
    
    # 避免除以零
    norm[norm < 1e-8] = 1.0
    output = output / norm
    
    return output


def main():
    # 配置参数
    checkpoint_path = "/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_weaker_student_2025-12-15-01h27m/checkpoints/best_model_150.tar"
    input_audio = "/home/sbplab/yuchen/GTCRN/SEtrain/experiments/outside_test/noisy.wav"
    output_audio = "/home/sbplab/yuchen/GTCRN/SEtrain/experiments/outside_test/enhanced.wav"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 确认文件存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 模型文件不存在: {checkpoint_path}")
        return
    
    if not os.path.exists(input_audio):
        print(f"❌ 错误: 输入音频不存在: {input_audio}")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)
    
    # 载入模型
    model = load_model(checkpoint_path, device=device)
    
    # 处理音频
    process_audio(
        model=model,
        audio_path=input_audio,
        output_path=output_audio,
        device=device,
        chunk_size=2.0,  # 2秒的片段
        sr=16000
    )
    
    print("\n" + "="*60)
    print("测试完成!")
    print(f"输入文件: {input_audio}")
    print(f"输出文件: {output_audio}")
    print("="*60)


if __name__ == "__main__":
    main()
