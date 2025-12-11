#!/usr/bin/env python3
"""
重新生成 CSV 列表，分離測試集
- 測試集：girl3, boy3 (640 個語音)
- 訓練+驗證集：boy1, boy2, girl1, girl2 (1280 個語音)
- 測試噪音：factory2.wav
- 訓練噪音：其他 7 種噪音
"""
import os
import pandas as pd
from pathlib import Path

def generate_split_csv_lists(base_dir='..'):
    """
    生成分離訓練集和測試集的 CSV 列表
    
    Args:
        base_dir: 基礎目錄（相對於 prepare_datasets）
    """
    base_path = Path(base_dir).resolve()
    speech_dir = base_path / 'speech_lib'
    
    print("=" * 60)
    print("Generating CSV lists with TEST SET separation")
    print("=" * 60)
    
    # ========== 1. 訓練/驗證集語音（排除 boy3, girl3）==========
    print("\n1. Generating TRAIN/VAL clean speech list...")
    train_val_files = []
    for speaker_dir in ['boy1', 'boy2', 'girl1', 'girl2']:
        speaker_path = speech_dir / speaker_dir
        if speaker_path.exists():
            files = sorted(speaker_path.glob('*.wav'))
            train_val_files.extend([str(f) for f in files])
    
    train_val_df = pd.DataFrame({'file_dir': train_val_files})
    train_val_csv = 'train_val_clean_dir.csv'
    train_val_df.to_csv(train_val_csv, index=False)
    print(f"   ✅ Created {train_val_csv}")
    print(f"   📊 Total train/val files: {len(train_val_files)}")
    
    # ========== 2. 測試集語音（僅 boy3, girl3）==========
    print("\n2. Generating TEST clean speech list...")
    test_files = []
    for speaker_dir in ['boy3', 'girl3']:
        speaker_path = speech_dir / speaker_dir
        if speaker_path.exists():
            files = sorted(speaker_path.glob('*.wav'))
            test_files.extend([str(f) for f in files])
    
    test_df = pd.DataFrame({'file_dir': test_files})
    test_csv = 'test_clean_dir.csv'
    test_df.to_csv(test_csv, index=False)
    print(f"   ✅ Created {test_csv}")
    print(f"   📊 Total test files: {len(test_files)}")
    
    # ========== 3. 訓練/驗證集噪音（排除 factory2）==========
    print("\n3. Generating TRAIN/VAL noise list...")
    noise_dir = speech_dir / 'noisex92_16k'
    
    if noise_dir.exists():
        # 取得所有噪音檔案並排除 factory2.wav
        all_noise_files = sorted(noise_dir.glob('*.wav'))
        train_val_noise_files = [str(f) for f in all_noise_files if 'factory2' not in f.name]
        
        print(f"   📊 Available noise files: {len(all_noise_files)}")
        print(f"   📊 Train/val noise files: {len(train_val_noise_files)}")
        
        # 重複噪音列表以匹配語音數量
        repeats = (len(train_val_files) // len(train_val_noise_files)) + 1
        train_val_noise_repeated = (train_val_noise_files * repeats)[:len(train_val_files)]
        
        train_val_noise_df = pd.DataFrame({'file_dir': train_val_noise_repeated})
        train_val_noise_csv = 'train_val_noise_dir.csv'
        train_val_noise_df.to_csv(train_val_noise_csv, index=False)
        print(f"   ✅ Created {train_val_noise_csv}")
        print(f"   📊 Train/val noise entries: {len(train_val_noise_repeated)}")
    
    # ========== 4. 測試集噪音（僅 factory2）==========
    print("\n4. Generating TEST noise list...")
    factory2_path = noise_dir / 'factory2.wav'
    
    if factory2_path.exists():
        # 重複 factory2.wav 以匹配測試語音數量
        test_noise_files = [str(factory2_path)] * len(test_files)
        
        test_noise_df = pd.DataFrame({'file_dir': test_noise_files})
        test_noise_csv = 'test_noise_dir.csv'
        test_noise_df.to_csv(test_noise_csv, index=False)
        print(f"   ✅ Created {test_noise_csv}")
        print(f"   📊 Test noise entries: {len(test_noise_files)}")
    
    # ========== 總結 ==========
    print("\n" + "=" * 60)
    print("✅ CSV generation completed!")
    print("=" * 60)
    print(f"Train/Val: {len(train_val_files)} clean + {len(train_val_noise_repeated)} noise")
    print(f"Test: {len(test_files)} clean + {len(test_noise_files)} noise")
    print("\nSpeakers distribution:")
    print(f"  Train/Val: boy1, boy2, girl1, girl2")
    print(f"  Test: boy3, girl3")
    print(f"\nNoise distribution:")
    print(f"  Train/Val: {len(train_val_noise_files)} types (excluding factory2)")
    print(f"  Test: factory2.wav only")


if __name__ == '__main__':
    generate_split_csv_lists()
