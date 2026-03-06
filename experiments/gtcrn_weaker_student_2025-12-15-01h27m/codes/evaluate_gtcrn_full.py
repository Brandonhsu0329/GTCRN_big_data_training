"""
GTCRN 完整評估腳本
- Inside: boy1 (訓練時見過)
- Outside: boy3, girl3 (訓練時未見過)
- 計算 STOI, PESQ (noisy 和 denoised 都算)
- 保存指定樣本的音檔
"""

import torch
import torchaudio
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from pesq import pesq
from pystoi import stoi
from omegaconf import OmegaConf

from models.gtcrn_end2end import GTCRN as Model

# ============================================================
# 配置
# ============================================================

CONFIG = {
    'model_path': '/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_custom_2025-12-10-01h59m/checkpoints/best_model_250.tar',
    'model_config': '/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_custom_2025-12-10-01h59m/config.yaml',
    'sample_rate': 16000,
    'output_dir': '/home/sbplab/yuchen/GTCRN/SEtrain/experiments/gtcrn_custom_2025-12-10-01h59m/evaluation_results',
    'speech_lib': '/home/sbplab/yuchen/GTCRN/SEtrain/speech_lib',

    # Inside 評估 (boy1 - 訓練時見過)
    'inside': {
        'speaker': 'boy1',
        'noises': ['babble', 'f16', 'factory1', 'leopard', 'pink', 'volvo', 'white'],
        'save_sentences': [1, 2],  # 保存第1、2句的音檔
    },

    # Outside 評估 (boy3 - 訓練時未見過)
    'outside_boy3': {
        'speaker': 'boy3',
        'noises': ['babble', 'f16', 'factory1', 'factory2', 'leopard', 'pink', 'volvo', 'white'],
        'save_sentences': [310, 320],  # 保存第310、320句的音檔
    },

    # Outside 評估 (girl3 - 訓練時未見過)
    'outside_girl3': {
        'speaker': 'girl3',
        'noises': ['babble', 'f16', 'factory1', 'factory2', 'leopard', 'pink', 'volvo', 'white'],
        'save_sentences': [310, 320],  # 保存第310、320句的音檔
    },

    'snr_levels': [0, 5, 10, 15, 20],
}

# ============================================================
# 工具函數
# ============================================================

def load_model(model_path, config_path, device):
    """載入訓練好的模型"""
    # 讀取配置
    cfg = OmegaConf.load(config_path)
    network_config = cfg['network_config']
    
    # 創建模型
    model = Model(**network_config).to(device)
    
    # 載入權重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"模型載入成功: {model_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    return model


def load_audio(audio_path, target_sr=16000):
    """載入並預處理音頻"""
    waveform, sr = torchaudio.load(audio_path)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    waveform = waveform.squeeze(0)
    
    # 歸一化
    waveform = waveform / (waveform.abs().max() + 1e-8)
    
    return waveform


def add_noise(clean, noise, snr_db):
    """添加噪音"""
    # 確保噪音夠長
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = noise.repeat(repeats)
    
    # 隨機起點
    if len(noise) > len(clean):
        start = np.random.randint(0, len(noise) - len(clean))
        noise = noise[start:start + len(clean)]
    
    # 計算縮放因子
    signal_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
    
    noisy = clean + scale * noise
    
    return noisy


def denoise(model, noisy_waveform, device):
    """使用模型去噪"""
    with torch.no_grad():
        noisy_input = noisy_waveform.unsqueeze(0).to(device)
        enhanced = model(noisy_input)
        enhanced = enhanced.cpu().squeeze(0)
    
    return enhanced


def calculate_metrics(clean, processed, sr=16000):
    """計算 STOI 和 PESQ"""
    clean_np = clean.numpy()
    processed_np = processed.numpy()
    
    # 確保長度相同
    min_len = min(len(clean_np), len(processed_np))
    clean_np = clean_np[:min_len]
    processed_np = processed_np[:min_len]
    
    # STOI
    try:
        stoi_score = stoi(clean_np, processed_np, sr, extended=False)
    except:
        stoi_score = float('nan')
    
    # PESQ
    try:
        pesq_score = pesq(sr, clean_np, processed_np, 'wb')
    except:
        pesq_score = float('nan')
    
    return stoi_score, pesq_score


# ============================================================
# 評估函數
# ============================================================

def evaluate_speaker(model, device, config, speaker_config, category_name):
    """評估單個說話者"""
    print(f"\n{'='*70}")
    print(f"評估 {category_name}: {speaker_config['speaker']}")
    print(f"{'='*70}")
    
    speech_lib = Path(config['speech_lib'])
    speaker_dir = speech_lib / speaker_config['speaker']
    noise_dir = speech_lib / 'noisex92_16k'
    output_dir = Path(config['output_dir']) / category_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入所有語音文件
    clean_files = sorted(list(speaker_dir.glob("*.wav")))
    print(f"找到 {len(clean_files)} 個語音文件")
    
    # 載入所有噪音
    noises = {}
    for noise_name in speaker_config['noises']:
        noise_path = noise_dir / f"{noise_name}.wav"
        if noise_path.exists():
            noises[noise_name] = load_audio(noise_path)
        else:
            print(f"警告: 找不到噪音文件 {noise_path}")
    print(f"載入 {len(noises)} 種噪音: {list(noises.keys())}")
    
    # 結果存儲
    results = {
        'all': {'noisy_stoi': [], 'noisy_pesq': [], 'denoised_stoi': [], 'denoised_pesq': []},
        'by_snr': {snr: {'noisy_stoi': [], 'noisy_pesq': [], 'denoised_stoi': [], 'denoised_pesq': []}
                   for snr in config['snr_levels']},
        'by_noise': {noise: {'noisy_stoi': [], 'noisy_pesq': [], 'denoised_stoi': [], 'denoised_pesq': []}
                     for noise in speaker_config['noises']},
    }
    
    # 要保存音檔的句子索引 (1-based to 0-based)
    save_indices = [i - 1 for i in speaker_config['save_sentences']]
    
    total = len(clean_files) * len(speaker_config['noises']) * len(config['snr_levels'])
    pbar = tqdm(total=total, desc=f"評估 {speaker_config['speaker']}")
    
    for file_idx, clean_file in enumerate(clean_files):
        clean = load_audio(clean_file)
        sentence_num = file_idx + 1  # 1-based
        
        for noise_name in speaker_config['noises']:
            if noise_name not in noises:
                pbar.update(len(config['snr_levels']))
                continue
                
            noise = noises[noise_name]
            
            for snr in config['snr_levels']:
                # 固定隨機種子以確保可重現
                np.random.seed(42 + file_idx * 1000 + hash(noise_name) % 1000 + snr)
                
                # 混合噪音
                noisy = add_noise(clean, noise, snr)
                
                # 去噪
                denoised = denoise(model, noisy, device)
                
                # 裁剪到相同長度
                min_len = min(len(clean), len(noisy), len(denoised))
                clean_trim = clean[:min_len]
                noisy_trim = noisy[:min_len]
                denoised_trim = denoised[:min_len]
                
                # 計算指標
                noisy_stoi, noisy_pesq = calculate_metrics(clean_trim, noisy_trim)
                denoised_stoi, denoised_pesq = calculate_metrics(clean_trim, denoised_trim)
                
                # 存儲結果
                results['all']['noisy_stoi'].append(noisy_stoi)
                results['all']['noisy_pesq'].append(noisy_pesq)
                results['all']['denoised_stoi'].append(denoised_stoi)
                results['all']['denoised_pesq'].append(denoised_pesq)
                
                results['by_snr'][snr]['noisy_stoi'].append(noisy_stoi)
                results['by_snr'][snr]['noisy_pesq'].append(noisy_pesq)
                results['by_snr'][snr]['denoised_stoi'].append(denoised_stoi)
                results['by_snr'][snr]['denoised_pesq'].append(denoised_pesq)
                
                results['by_noise'][noise_name]['noisy_stoi'].append(noisy_stoi)
                results['by_noise'][noise_name]['noisy_pesq'].append(noisy_pesq)
                results['by_noise'][noise_name]['denoised_stoi'].append(denoised_stoi)
                results['by_noise'][noise_name]['denoised_pesq'].append(denoised_pesq)
                
                # 保存指定的音檔
                if file_idx in save_indices:
                    audio_dir = output_dir / f"sentence_{sentence_num}" / noise_name / f"snr_{snr}"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    
                    torchaudio.save(audio_dir / "clean.wav", clean_trim.unsqueeze(0), 16000)
                    torchaudio.save(audio_dir / "noisy_input.wav", noisy_trim.unsqueeze(0), 16000)
                    torchaudio.save(audio_dir / "denoised_output.wav", denoised_trim.unsqueeze(0), 16000)
                
                pbar.update(1)
    
    pbar.close()
    
    # 計算平均值
    def calc_mean(values):
        valid = [v for v in values if not np.isnan(v)]
        return np.mean(valid) if valid else float('nan')
    
    summary = {
        'total': {
            'noisy_stoi': calc_mean(results['all']['noisy_stoi']),
            'noisy_pesq': calc_mean(results['all']['noisy_pesq']),
            'denoised_stoi': calc_mean(results['all']['denoised_stoi']),
            'denoised_pesq': calc_mean(results['all']['denoised_pesq']),
        },
        'by_snr': {},
        'by_noise': {},
    }
    
    for snr in config['snr_levels']:
        summary['by_snr'][snr] = {
            'noisy_stoi': calc_mean(results['by_snr'][snr]['noisy_stoi']),
            'noisy_pesq': calc_mean(results['by_snr'][snr]['noisy_pesq']),
            'denoised_stoi': calc_mean(results['by_snr'][snr]['denoised_stoi']),
            'denoised_pesq': calc_mean(results['by_snr'][snr]['denoised_pesq']),
        }
    
    for noise_name in speaker_config['noises']:
        summary['by_noise'][noise_name] = {
            'noisy_stoi': calc_mean(results['by_noise'][noise_name]['noisy_stoi']),
            'noisy_pesq': calc_mean(results['by_noise'][noise_name]['noisy_pesq']),
            'denoised_stoi': calc_mean(results['by_noise'][noise_name]['denoised_stoi']),
            'denoised_pesq': calc_mean(results['by_noise'][noise_name]['denoised_pesq']),
        }
    
    return summary


def print_results(all_results):
    """打印所有結果表格"""
    print("\n")
    print("=" * 100)
    print("評估結果總表")
    print("=" * 100)
    
    for category, summary in all_results.items():
        print(f"\n{'='*80}")
        print(f"【{category}】")
        print(f"{'='*80}")
        
        # 總平均
        print(f"\n--- 總平均 ---")
        print(f"{'指標':<15} {'Noisy':<15} {'Denoised':<15} {'改善':<15}")
        print("-" * 60)
        
        stoi_imp = summary['total']['denoised_stoi'] - summary['total']['noisy_stoi']
        pesq_imp = summary['total']['denoised_pesq'] - summary['total']['noisy_pesq']
        
        print(f"{'STOI':<15} {summary['total']['noisy_stoi']:<15.4f} {summary['total']['denoised_stoi']:<15.4f} {stoi_imp:+.4f}")
        print(f"{'PESQ':<15} {summary['total']['noisy_pesq']:<15.4f} {summary['total']['denoised_pesq']:<15.4f} {pesq_imp:+.4f}")
        
        # 按 SNR
        print(f"\n--- 按 SNR 細分 ---")
        print(f"{'SNR':<8} {'Noisy STOI':<12} {'Den STOI':<12} {'Noisy PESQ':<12} {'Den PESQ':<12} {'STOI Δ':<10} {'PESQ Δ':<10}")
        print("-" * 76)
        for snr in sorted(summary['by_snr'].keys()):
            data = summary['by_snr'][snr]
            stoi_d = data['denoised_stoi'] - data['noisy_stoi']
            pesq_d = data['denoised_pesq'] - data['noisy_pesq']
            print(f"{snr:<8} {data['noisy_stoi']:<12.4f} {data['denoised_stoi']:<12.4f} {data['noisy_pesq']:<12.4f} {data['denoised_pesq']:<12.4f} {stoi_d:+.4f}    {pesq_d:+.4f}")
        
        # 按噪音
        print(f"\n--- 按噪音類型細分 ---")
        print(f"{'噪音':<12} {'Noisy STOI':<12} {'Den STOI':<12} {'Noisy PESQ':<12} {'Den PESQ':<12} {'STOI Δ':<10} {'PESQ Δ':<10}")
        print("-" * 80)
        for noise_name in sorted(summary['by_noise'].keys()):
            data = summary['by_noise'][noise_name]
            stoi_d = data['denoised_stoi'] - data['noisy_stoi']
            pesq_d = data['denoised_pesq'] - data['noisy_pesq']
            print(f"{noise_name:<12} {data['noisy_stoi']:<12.4f} {data['denoised_stoi']:<12.4f} {data['noisy_pesq']:<12.4f} {data['denoised_pesq']:<12.4f} {stoi_d:+.4f}    {pesq_d:+.4f}")


def main():
    print("=" * 70)
    print("GTCRN 完整評估")
    print("=" * 70)
    
    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建輸出目錄
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # 載入模型
    model = load_model(CONFIG['model_path'], CONFIG['model_config'], device)
    
    # 評估三組
    all_results = {}
    
    # Inside (boy1)
    all_results['Inside (boy1)'] = evaluate_speaker(
        model, device, CONFIG, CONFIG['inside'], 'inside_boy1'
    )
    
    # Outside (boy3)
    all_results['Outside (boy3)'] = evaluate_speaker(
        model, device, CONFIG, CONFIG['outside_boy3'], 'outside_boy3'
    )
    
    # Outside (girl3)
    all_results['Outside (girl3)'] = evaluate_speaker(
        model, device, CONFIG, CONFIG['outside_girl3'], 'outside_girl3'
    )
    
    # 打印結果
    print_results(all_results)
    
    # 保存結果到 JSON
    results_path = output_dir / 'full_evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n結果已保存到: {results_path}")
    
    # 打印音檔保存位置
    print(f"\n音檔已保存到:")
    print(f"  Inside (boy1 第1、2句): {output_dir}/inside_boy1/")
    print(f"  Outside (boy3 第310、320句): {output_dir}/outside_boy3/")
    print(f"  Outside (girl3 第310、320句): {output_dir}/outside_girl3/")
    
    print("\n" + "=" * 70)
    print("評估完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
