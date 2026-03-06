#!/usr/bin/env python3
"""
音频信息查看脚本
快速查看音频文件的基本信息
"""
import soundfile as sf
import numpy as np
import os


def analyze_audio(audio_path):
    """分析音频文件"""
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return
    
    # 读取音频
    audio, sr = sf.read(audio_path, dtype='float32')
    
    # 计算基本统计信息
    duration = len(audio) / sr
    max_val = np.abs(audio).max()
    mean_val = np.mean(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    
    # 文件大小
    file_size = os.path.getsize(audio_path) / 1024  # KB
    
    print(f"\n{'='*60}")
    print(f"文件: {os.path.basename(audio_path)}")
    print(f"{'='*60}")
    print(f"采样率:     {sr} Hz")
    print(f"时长:       {duration:.2f} 秒")
    print(f"采样点数:   {len(audio)}")
    print(f"文件大小:   {file_size:.1f} KB")
    print(f"最大幅度:   {max_val:.4f}")
    print(f"平均幅度:   {mean_val:.4f}")
    print(f"RMS:        {rms:.4f}")
    print(f"RMS (dB):   {20*np.log10(rms + 1e-8):.2f} dB")
    
    # 检查削波
    if max_val > 0.99:
        print(f"⚠️  警告: 音频可能存在削波 (最大值: {max_val:.4f})")


def main():
    # 分析所有音频文件
    audio_dir = "/home/sbplab/yuchen/GTCRN/SEtrain/experiments/outside_test"
    
    audio_files = ['noisy.wav', 'enhanced.wav', 'clean.wav']
    
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        analyze_audio(audio_path)
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
