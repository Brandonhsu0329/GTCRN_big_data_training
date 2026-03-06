#!/usr/bin/env python3
"""
音频质量评估脚本
评估降噪后的音频质量（需要clean参考音频）
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


def calculate_si_sdr(reference, estimation):
    """
    计算 Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    
    Args:
        reference: 参考信号 (干净语音)
        estimation: 估计信号 (增强语音)
    
    Returns:
        SI-SDR值 (dB)
    """
    # 确保信号长度相同
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    # 移除均值
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # 计算最优缩放因子
    alpha = np.dot(estimation, reference) / (np.dot(reference, reference) + 1e-8)
    
    # 投影
    projection = alpha * reference
    
    # 计算噪声
    noise = estimation - projection
    
    # 计算 SI-SDR
    si_sdr = 10 * np.log10(
        (np.dot(projection, projection) + 1e-8) / 
        (np.dot(noise, noise) + 1e-8)
    )
    
    return si_sdr


def calculate_snr(clean, noisy_or_enhanced):
    """
    计算信噪比 (SNR)
    
    Args:
        clean: 干净语音
        noisy_or_enhanced: 带噪或增强后的语音
    
    Returns:
        SNR值 (dB)
    """
    # 确保信号长度相同
    min_len = min(len(clean), len(noisy_or_enhanced))
    clean = clean[:min_len]
    noisy_or_enhanced = noisy_or_enhanced[:min_len]
    
    # 计算噪声
    noise = noisy_or_enhanced - clean
    
    # 计算SNR
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    snr = 10 * np.log10((signal_power + 1e-8) / (noise_power + 1e-8))
    
    return snr


def calculate_pesq(reference_path, degraded_path, sr=16000):
    """
    计算 PESQ (需要安装 pesq 库)
    
    Args:
        reference_path: 参考音频路径
        degraded_path: 待评估音频路径
        sr: 采样率
    
    Returns:
        PESQ分数 (或 None 如果无法计算)
    """
    try:
        from pesq import pesq
        
        reference, _ = sf.read(reference_path)
        degraded, _ = sf.read(degraded_path)
        
        # PESQ 需要16kHz
        if sr == 16000:
            mode = 'wb'  # wideband
        elif sr == 8000:
            mode = 'nb'  # narrowband
        else:
            print(f"   ⚠️  PESQ 只支持 8kHz 或 16kHz，当前采样率: {sr} Hz")
            return None
        
        # 确保长度相同
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        score = pesq(sr, reference, degraded, mode)
        return score
    
    except ImportError:
        print("   ⚠️  未安装 pesq 库，跳过 PESQ 计算")
        print("   安装方法: pip install pesq")
        return None
    except Exception as e:
        print(f"   ⚠️  PESQ 计算失败: {e}")
        return None


def calculate_stoi(reference_path, degraded_path, sr=16000):
    """
    计算 STOI (Short-Time Objective Intelligibility)
    
    Args:
        reference_path: 参考音频路径
        degraded_path: 待评估音频路径
        sr: 采样率
    
    Returns:
        STOI分数 (或 None 如果无法计算)
    """
    try:
        from pystoi import stoi
        
        reference, _ = sf.read(reference_path)
        degraded, _ = sf.read(degraded_path)
        
        # 确保长度相同
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        score = stoi(reference, degraded, sr, extended=False)
        return score
    
    except ImportError:
        print("   ⚠️  未安装 pystoi 库，跳过 STOI 计算")
        print("   安装方法: pip install pystoi")
        return None
    except Exception as e:
        print(f"   ⚠️  STOI 计算失败: {e}")
        return None


def evaluate_audio(clean_path, noisy_path, enhanced_path, sr=16000):
    """
    评估音频质量
    
    Args:
        clean_path: 干净音频路径 (参考)
        noisy_path: 带噪音频路径
        enhanced_path: 增强后音频路径
        sr: 采样率
    """
    print("=" * 60)
    print("音频质量评估")
    print("=" * 60)
    
    # 读取音频
    try:
        clean, fs_clean = sf.read(clean_path, dtype='float32')
        noisy, fs_noisy = sf.read(noisy_path, dtype='float32')
        enhanced, fs_enhanced = sf.read(enhanced_path, dtype='float32')
    except Exception as e:
        print(f"❌ 读取音频文件失败: {e}")
        return
    
    # 检查采样率
    if fs_clean != sr or fs_noisy != sr or fs_enhanced != sr:
        print(f"⚠️  警告: 采样率不一致")
        print(f"   Clean: {fs_clean} Hz, Noisy: {fs_noisy} Hz, Enhanced: {fs_enhanced} Hz")
    
    print(f"\n文件信息:")
    print(f"  Clean:    {len(clean)} 采样点 ({len(clean)/fs_clean:.2f} 秒)")
    print(f"  Noisy:    {len(noisy)} 采样点 ({len(noisy)/fs_noisy:.2f} 秒)")
    print(f"  Enhanced: {len(enhanced)} 采样点 ({len(enhanced)/fs_enhanced:.2f} 秒)")
    
    # 确保所有音频长度相同
    min_len = min(len(clean), len(noisy), len(enhanced))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]
    
    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)
    
    # SI-SDR
    print("\n📊 SI-SDR (Scale-Invariant Signal-to-Distortion Ratio):")
    si_sdr_noisy = calculate_si_sdr(clean, noisy)
    si_sdr_enhanced = calculate_si_sdr(clean, enhanced)
    si_sdr_improvement = si_sdr_enhanced - si_sdr_noisy
    print(f"   Noisy:    {si_sdr_noisy:.2f} dB")
    print(f"   Enhanced: {si_sdr_enhanced:.2f} dB")
    print(f"   ✨ 提升:   {si_sdr_improvement:+.2f} dB")
    
    # SNR
    print("\n📊 SNR (Signal-to-Noise Ratio):")
    snr_noisy = calculate_snr(clean, noisy)
    snr_enhanced = calculate_snr(clean, enhanced)
    snr_improvement = snr_enhanced - snr_noisy
    print(f"   Noisy:    {snr_noisy:.2f} dB")
    print(f"   Enhanced: {snr_enhanced:.2f} dB")
    print(f"   ✨ 提升:   {snr_improvement:+.2f} dB")
    
    # PESQ
    print("\n📊 PESQ (Perceptual Evaluation of Speech Quality):")
    pesq_noisy = calculate_pesq(clean_path, noisy_path, sr)
    pesq_enhanced = calculate_pesq(clean_path, enhanced_path, sr)
    
    if pesq_noisy is not None and pesq_enhanced is not None:
        pesq_improvement = pesq_enhanced - pesq_noisy
        print(f"   Noisy:    {pesq_noisy:.3f}")
        print(f"   Enhanced: {pesq_enhanced:.3f}")
        print(f"   ✨ 提升:   {pesq_improvement:+.3f}")
    
    # STOI
    print("\n📊 STOI (Short-Time Objective Intelligibility):")
    stoi_noisy = calculate_stoi(clean_path, noisy_path, sr)
    stoi_enhanced = calculate_stoi(clean_path, enhanced_path, sr)
    
    if stoi_noisy is not None and stoi_enhanced is not None:
        stoi_improvement = stoi_enhanced - stoi_noisy
        print(f"   Noisy:    {stoi_noisy:.4f}")
        print(f"   Enhanced: {stoi_enhanced:.4f}")
        print(f"   ✨ 提升:   {stoi_improvement:+.4f}")
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    
    # 保存结果到文件
    results_file = os.path.join(os.path.dirname(enhanced_path), "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("音频质量评估结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Clean:    {clean_path}\n")
        f.write(f"Noisy:    {noisy_path}\n")
        f.write(f"Enhanced: {enhanced_path}\n\n")
        
        f.write("SI-SDR (dB):\n")
        f.write(f"  Noisy:    {si_sdr_noisy:.2f}\n")
        f.write(f"  Enhanced: {si_sdr_enhanced:.2f}\n")
        f.write(f"  提升:      {si_sdr_improvement:+.2f}\n\n")
        
        f.write("SNR (dB):\n")
        f.write(f"  Noisy:    {snr_noisy:.2f}\n")
        f.write(f"  Enhanced: {snr_enhanced:.2f}\n")
        f.write(f"  提升:      {snr_improvement:+.2f}\n\n")
        
        if pesq_noisy is not None and pesq_enhanced is not None:
            f.write("PESQ:\n")
            f.write(f"  Noisy:    {pesq_noisy:.3f}\n")
            f.write(f"  Enhanced: {pesq_enhanced:.3f}\n")
            f.write(f"  提升:      {pesq_improvement:+.3f}\n\n")
        
        if stoi_noisy is not None and stoi_enhanced is not None:
            f.write("STOI:\n")
            f.write(f"  Noisy:    {stoi_noisy:.4f}\n")
            f.write(f"  Enhanced: {stoi_enhanced:.4f}\n")
            f.write(f"  提升:      {stoi_improvement:+.4f}\n\n")
    
    print(f"\n结果已保存到: {results_file}")


def main():
    # 配置参数 - 如果有clean参考音频，请修改这里
    clean_path = "/home/sbplab/yuchen/GTCRN/SEtrain/experiments/outside_test/clean.wav"  # 需要提供clean音频
    noisy_path = "/home/sbplab/yuchen/GTCRN/SEtrain/experiments/outside_test/noisy.wav"
    enhanced_path = "/home/sbplab/yuchen/GTCRN/SEtrain/experiments/outside_test/enhanced.wav"
    
    # 检查文件是否存在
    if not os.path.exists(clean_path):
        print(f"❌ 错误: Clean音频文件不存在: {clean_path}")
        print("   如果没有clean参考音频，无法进行完整评估")
        print("   你可以:")
        print("   1. 提供clean音频文件")
        print("   2. 或者只进行主观听感评估")
        return
    
    if not os.path.exists(noisy_path):
        print(f"❌ 错误: Noisy音频文件不存在: {noisy_path}")
        return
    
    if not os.path.exists(enhanced_path):
        print(f"❌ 错误: Enhanced音频文件不存在: {enhanced_path}")
        print("   请先运行 test_single_audio.py 生成增强后的音频")
        return
    
    # 评估音频
    evaluate_audio(clean_path, noisy_path, enhanced_path, sr=16000)


if __name__ == "__main__":
    main()
