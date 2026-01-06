import os
import csv
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch

def add_noise(clean, noise, snr_db):
    """根據指定 SNR 添加噪音到乾淨語音"""
    # 確保噪音夠長
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    
    # 隨機起點
    if len(noise) > len(clean):
        start = np.random.randint(0, len(noise) - len(clean))
        noise = noise[start:start + len(clean)]
    
    # 計算縮放因子
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(signal_power / (snr_linear * noise_power + 1e-10))
    
    noisy = clean + scale * noise
    
    return noisy, clean

def process_csv_and_generate_audio(csv_file, output_dir, data_type):
    """讀取 CSV 並生成混合音訊"""
    print(f"\n{'='*60}")
    print(f"處理 {data_type} 資料")
    print(f"{'='*60}")
    
    # 創建輸出目錄
    noisy_dir = os.path.join(output_dir, f"{data_type}_data", f"{data_type}_noisy")
    clean_dir = os.path.join(output_dir, f"{data_type}_data", f"{data_type}_clean")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    # 讀取 CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"總共 {len(rows)} 個檔案需要生成")
    print(f"輸出目錄: {noisy_dir}")
    
    # 使用 tqdm 顯示進度
    for row in tqdm(rows, desc=f"生成 {data_type} 音訊"):
        file_name = row['file_name']
        clean_path = row['clean']
        noise_path = row['noise']
        snr = float(row['snr'])
        
        try:
            # 讀取乾淨語音和噪音
            clean_audio, sr_clean = sf.read(clean_path, dtype='float32')
            noise_audio, sr_noise = sf.read(noise_path, dtype='float32')
            
            # 確保採樣率一致（應該都是 16kHz）
            if sr_clean != 16000:
                print(f"警告: {clean_path} 採樣率為 {sr_clean}, 預期為 16000")
            if sr_noise != 16000:
                print(f"警告: {noise_path} 採樣率為 {sr_noise}, 預期為 16000")
            
            # 混合音訊
            noisy_audio, clean_audio = add_noise(clean_audio, noise_audio, snr)
            
            # 正規化到 [-1, 1] 範圍
            max_val = np.abs(noisy_audio).max()
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val
                clean_audio = clean_audio / max_val
            
            # 儲存檔案
            noisy_output = os.path.join(noisy_dir, file_name)
            clean_output = os.path.join(clean_dir, file_name)
            
            sf.write(noisy_output, noisy_audio, 16000)
            sf.write(clean_output, clean_audio, 16000)
            
        except Exception as e:
            print(f"\n錯誤處理 {file_name}: {str(e)}")
            continue
    
    print(f"✓ 完成 {data_type} 資料生成")
    print(f"  - Noisy: {noisy_dir}")
    print(f"  - Clean: {clean_dir}")

def main():
    base_path = "/home/sbplab/yuchen/GTCRN/SEtrain/datasets2"
    
    # 設定隨機種子以確保可重現性
    np.random.seed(42)
    
    print("="*60)
    print("開始生成混合音訊資料集")
    print("="*60)
    
    # 處理訓練資料
    train_csv = os.path.join(base_path, "train_data", "train_INFO.csv")
    if os.path.exists(train_csv):
        process_csv_and_generate_audio(train_csv, base_path, "train")
    else:
        print(f"找不到 {train_csv}")
    
    # 處理驗證資料
    val_csv = os.path.join(base_path, "val_data", "val_INFO.csv")
    if os.path.exists(val_csv):
        process_csv_and_generate_audio(val_csv, base_path, "val")
    else:
        print(f"找不到 {val_csv}")
    
    # 處理測試資料
    test_csv = os.path.join(base_path, "test_data", "test_INFO.csv")
    if os.path.exists(test_csv):
        process_csv_and_generate_audio(test_csv, base_path, "test")
    else:
        print(f"找不到 {test_csv}")
    
    print("\n" + "="*60)
    print("全部完成！")
    print("="*60)
    print("\n資料集結構：")
    print(f"{base_path}/")
    print("├── train_data/")
    print("│   ├── train_INFO.csv")
    print("│   ├── train_noisy/  (含噪音的音訊)")
    print("│   └── train_clean/  (乾淨的音訊)")
    print("├── val_data/")
    print("│   ├── val_INFO.csv")
    print("│   ├── val_noisy/")
    print("│   └── val_clean/")
    print("└── test_data/")
    print("    ├── test_INFO.csv")
    print("    ├── test_noisy/")
    print("    └── test_clean/")

if __name__ == "__main__":
    main()
