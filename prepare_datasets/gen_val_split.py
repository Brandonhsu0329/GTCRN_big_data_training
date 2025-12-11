import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def pad_or_truncate(audio, target_length):
    """
    將音訊補零或裁切到目標長度
    
    Args:
        audio: 輸入音訊
        target_length: 目標長度（樣本數）
    
    Returns:
        處理後的音訊
    """
    current_length = len(audio)
    
    if current_length < target_length:
        # 補零
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant')
    elif current_length > target_length:
        # 隨機裁切
        start = np.random.randint(0, current_length - target_length + 1)
        audio = audio[start:start + target_length]
    
    return audio


def mk_mixture(clean, noise, snr, target_length, eps=1e-8):
    """
    混合語音和噪音，並確保輸出長度一致（無混響版本）
    
    Args:
        clean: 乾淨語音
        noise: 噪音
        snr: 信噪比
        target_length: 目標長度（樣本數）
        eps: 小數值防止除零
    
    Returns:
        mixture: 混合後的含噪語音
        target: 目標乾淨語音
    """
    # 確保所有信號長度一致
    clean = pad_or_truncate(clean, target_length)
    noise = pad_or_truncate(noise, target_length)
    
    # 正規化乾淨語音
    amp = 0.5 * np.random.rand() + 0.01
    clean = amp * clean / (np.max(np.abs(clean)) + eps)
    
    # 根據 SNR 混合噪音
    norm_noise = noise * np.sqrt(np.sum(clean ** 2) + eps) / np.sqrt(np.sum(noise ** 2) + eps)
    alpha = 10**(-snr * 1.5 / 20)
    
    mix = clean + alpha * norm_noise
    
    # 正規化到 [-1, 1]
    M = max(np.max(abs(mix)), np.max(abs(clean)), np.max(abs(alpha * norm_noise))) + eps
    if M > 1.0:    
        mix = mix / M
        clean = clean / M

    return mix, clean


if __name__ == "__main__":
    np.random.seed(42)
    
    # ========== 配置參數 ==========
    total_files = 1280  # 訓練+驗證總數
    train_ratio = 0.9   
    num_train = int(total_files * train_ratio)  # 1152 個訓練
    num_val = total_files - num_train  # 128 個驗證
    nfill = len(str(total_files))
    
    fs = 16000
    wav_len = 2  # 2 秒
    target_length = fs * wav_len  # 32000 樣本
    snr_range = [-5, 15]
    
    # 輸出到 SEtrain/datasets 目錄
    save_root = '../datasets/val_data/'
    
    # 輸入 CSV 文件
    data_root = './'
    clean_csv = os.path.join(data_root, 'train_val_clean_dir.csv')
    noise_csv = os.path.join(data_root, 'train_val_noise_dir.csv')
    
    # ========== 讀取文件列表 ==========
    print("=" * 60)
    print("Loading file lists for VALIDATION set (10% of train/val)...")
    print("=" * 60)
    
    # 取後 10% 作為驗證集
    clean_list_all = pd.read_csv(clean_csv)['file_dir'].tolist()
    noise_list_all = pd.read_csv(noise_csv)['file_dir'].tolist()
    
    clean_list = clean_list_all[num_train:num_train+num_val]
    noise_list = noise_list_all[num_train:num_train+num_val]
    snr_list = np.random.uniform(snr_range[0], snr_range[1], size=num_val)
    
    print(f"Clean files: {len(clean_list)} (indices {num_train}-{num_train+num_val-1})")
    print(f"Noise files: {len(noise_list)}")
    print(f"Target length: {wav_len} seconds ({target_length} samples)")
    print(f"Speakers: boy1, boy2, girl1, girl2 only")
    print(f"Noise types: 7 types (excluding factory2)")

    
    # ========== 創建輸出目錄 ==========
    os.makedirs(os.path.join(save_root, 'train_noisy'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'train_clean'), exist_ok=True)
    
    # ========== 保存配置信息 ==========
    info = pd.DataFrame([str(idx+1).zfill(nfill)+'.wav' for idx in range(num_val)], 
                        columns=['file_name'])
    info['clean'] = clean_list
    info['noise'] = noise_list
    info['snr'] = snr_list
    
    info.to_csv(os.path.join(save_root, 'train_INFO.csv'), index=None)
    
    # ========== 生成驗證數據 ==========
    print("\n" + "=" * 60)
    print(f"Generating {num_val} validation samples (without reverberation)...")
    print("=" * 60)
    
    for idx in tqdm(range(num_val)):
        try:
            # 讀取文件
            clean, sr_clean = sf.read(clean_list[idx], dtype='float32')
            noise, sr_noise = sf.read(noise_list[idx], dtype='float32')
            
            # 確保採樣率一致
            assert sr_clean == fs, f"Clean speech sample rate {sr_clean} != {fs}"
            assert sr_noise == fs, f"Noise sample rate {sr_noise} != {fs}"
            
            # 直接混合噪音（無混響處理）
            mixture, target = mk_mixture(clean, noise, snr_list[idx], 
                                        target_length, eps=1e-8)
            
            # 儲存
            output_noisy = os.path.join(save_root, 'train_noisy', 
                                       str(idx+1).zfill(nfill)+'.wav')
            output_clean = os.path.join(save_root, 'train_clean', 
                                       str(idx+1).zfill(nfill)+'.wav')
            
            sf.write(output_noisy, mixture, fs)
            sf.write(output_clean, target, fs)
            
        except Exception as e:
            print(f"\nError processing sample {idx+1}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ VALIDATION dataset generation completed!")
    print("=" * 60)
    print(f"Output directory: {os.path.abspath(save_root)}")
    print(f"Total samples: {num_val}")
    print(f"Sample length: {wav_len} seconds")
    print(f"Noisy files: {save_root}train_noisy/")
    print(f"Clean files: {save_root}train_clean/")
    print(f"Info file: {save_root}train_INFO.csv")
