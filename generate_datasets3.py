import os
import csv
import glob

# 基本路徑設定
base_path = "/home/sbplab/yuchen/GTCRN/SEtrain"
speech_lib_path = os.path.join(base_path, "speech_lib_bigger")
datasets3_path = os.path.join(base_path, "datasets3")

# 創建 datasets3 目錄結構
os.makedirs(os.path.join(datasets3_path, "train_data"), exist_ok=True)
os.makedirs(os.path.join(datasets3_path, "val_data"), exist_ok=True)
os.makedirs(os.path.join(datasets3_path, "test_data"), exist_ok=True)

# 定義配置
train_config = {
    'speakers': ['boy1', 'boy2', 'boy3', 'boy4', 'boy5', 'girl1', 'girl2', 'girl3', 'girl4', 'girl6'],
    'noises': ['babble.wav', 'destroyerengine.wav', 'factory1.wav', 'volvo.wav', 
               'buccaneer1.wav', 'destroyerops.wav', 'factory2.wav', 'white.wav',
               'buccaneer2.wav', 'f16.wav', 'hfchannel.wav', 'pink.wav'],
    'snrs': [-5, 0, 5, 10, 15]
}

val_config = {
    'speakers': ['boy6'],
    'noises': ['factory2.wav'],
    'snrs': [-5, 0, 5, 10, 15]
}

test_config = {
    'speakers': ['boy6', 'girl7'],
    'noises': ['n45.wav', 'factory2.wav'],
    'snrs': [-5, 0, 5, 10, 15]
}

def get_clean_files(speaker):
    """獲取指定說話者的所有乾淨語音檔案"""
    speaker_path = os.path.join(speech_lib_path, speaker)
    clean_files = sorted(glob.glob(os.path.join(speaker_path, "*.wav")))
    return clean_files

def generate_csv(config, output_file, start_index=1):
    """生成 CSV 檔案"""
    rows = []
    file_index = start_index
    
    # 對於每個說話者
    for speaker in config['speakers']:
        clean_files = get_clean_files(speaker)
        print(f"處理 {speaker}: 找到 {len(clean_files)} 個乾淨語音檔案")
        
        # 對於每個乾淨語音檔案
        for clean_file in clean_files:
            # 對於每種噪音
            for noise in config['noises']:
                noise_path = os.path.join(speech_lib_path, "noisex92_16k", noise)
                
                # 對於每個 SNR 值
                for snr in config['snrs']:
                    file_name = f"{file_index:04d}.wav"
                    rows.append({
                        'file_name': file_name,
                        'clean': clean_file,
                        'noise': noise_path,
                        'snr': snr
                    })
                    file_index += 1
    
    # 寫入 CSV 檔案
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'clean', 'noise', 'snr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"已生成 {output_file}")
    print(f"總共 {len(rows)} 筆資料\n")
    
    return file_index

# 生成 train_INFO.csv
print("="*70)
print("生成 datasets3 - 使用 speech_lib_bigger")
print("="*70)
print("\n=== 生成訓練資料 ===")
train_output = os.path.join(datasets3_path, "train_data", "train_INFO.csv")
next_index = generate_csv(train_config, train_output, start_index=1)

# 生成 val_INFO.csv
print("=== 生成驗證資料 ===")
val_output = os.path.join(datasets3_path, "val_data", "val_INFO.csv")
next_index = generate_csv(val_config, val_output, start_index=next_index)

# 生成 test_INFO.csv
print("=== 生成測試資料 ===")
test_output = os.path.join(datasets3_path, "test_data", "test_INFO.csv")
generate_csv(test_config, test_output, start_index=next_index)

print("\n" + "="*70)
print("完成！")
print("="*70)
print(f"\n所有檔案已生成於: {datasets3_path}")
print(f"- 訓練資料: {train_output}")
print(f"- 驗證資料: {val_output}")
print(f"- 測試資料: {test_output}")

# 顯示統計資訊
print("\n" + "="*70)
print("資料統計")
print("="*70)
train_total = len(train_config['speakers']) * 320 * len(train_config['noises']) * len(train_config['snrs'])
val_total = len(val_config['speakers']) * 320 * len(val_config['noises']) * len(val_config['snrs'])
test_total = len(test_config['speakers']) * 320 * len(test_config['noises']) * len(test_config['snrs'])

print(f"訓練資料: {len(train_config['speakers'])} 說話者 × 320個語音 × {len(train_config['noises'])} 噪音 × {len(train_config['snrs'])} SNR = {train_total:,} 筆")
print(f"驗證資料: {len(val_config['speakers'])} 說話者 × 320個語音 × {len(val_config['noises'])} 噪音 × {len(val_config['snrs'])} SNR = {val_total:,} 筆")
print(f"測試資料: {len(test_config['speakers'])} 說話者 × 320個語音 × {len(test_config['noises'])} 噪音 × {len(test_config['snrs'])} SNR = {test_total:,} 筆")
print(f"\n總計: {train_total + val_total + test_total:,} 筆混合音訊")

print("\n使用的噪音類型：")
print(f"  訓練: {', '.join(train_config['noises'])}")
print(f"  驗證: {', '.join(val_config['noises'])}")
print(f"  測試: {', '.join(test_config['noises'])}")
