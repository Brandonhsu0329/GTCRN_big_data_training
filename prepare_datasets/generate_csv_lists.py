#!/usr/bin/env python3
"""
生成訓練數據的 CSV 列表文件
- 乾淨語音列表
- 噪音列表
- RIR 列表
"""
import os
import pandas as pd
from pathlib import Path

def generate_csv_lists(base_dir='..'):
    """
    生成所有需要的 CSV 列表
    
    Args:
        base_dir: 基礎目錄（相對於 prepare_datasets）
    """
    base_path = Path(base_dir).resolve()
    
    print("=" * 60)
    print("Generating CSV lists for training data")
    print("=" * 60)
    
    # 1. 生成乾淨語音列表
    print("\n1. Generating clean speech list...")
    clean_files = []
    speech_dir = base_path / 'speech_lib'
    
    for speaker_dir in ['boy1', 'boy2', 'boy3', 'girl1', 'girl2', 'girl3']:
        speaker_path = speech_dir / speaker_dir
        if speaker_path.exists():
            files = sorted(speaker_path.glob('*.wav'))
            clean_files.extend([str(f) for f in files])
    
    clean_df = pd.DataFrame({'file_dir': clean_files})
    clean_csv_path = 'train_clean_dir.csv'
    clean_df.to_csv(clean_csv_path, index=False)
    print(f"   ✅ Created {clean_csv_path}")
    print(f"   📊 Total clean files: {len(clean_files)}")
    
    # 2. 生成噪音列表（重採樣後的）
    print("\n2. Generating noise list...")
    noise_dir = speech_dir / 'noisex92_16k'
    
    if noise_dir.exists():
        noise_files = sorted(noise_dir.glob('*.wav'))
        noise_files = [str(f) for f in noise_files]
        
        # 重複噪音列表以匹配語音數量
        repeats = (len(clean_files) // len(noise_files)) + 1
        noise_files_repeated = (noise_files * repeats)[:len(clean_files)]
        
        noise_df = pd.DataFrame({'file_dir': noise_files_repeated})
        noise_csv_path = 'train_noise_dir.csv'
        noise_df.to_csv(noise_csv_path, index=False)
        print(f"   ✅ Created {noise_csv_path}")
        print(f"   📊 Unique noise files: {len(noise_files)}")
        print(f"   📊 Total noise entries: {len(noise_files_repeated)}")
    else:
        print(f"   ⚠️  Noise directory not found: {noise_dir}")
        print(f"   Please run resample_noise.py first!")
    
    # 3. 生成 RIR 列表
    print("\n3. Generating RIR list...")
    rir_dir = speech_dir / 'rirs'
    
    if rir_dir.exists():
        rir_files = sorted(rir_dir.glob('*.wav'))
        rir_files = [str(f) for f in rir_files]
        
        # 重複 RIR 列表以匹配語音數量
        repeats = (len(clean_files) // len(rir_files)) + 1
        rir_files_repeated = (rir_files * repeats)[:len(clean_files)]
        
        rir_df = pd.DataFrame({'file_dir': rir_files_repeated})
        rir_csv_path = 'train_rir_dir.csv'
        rir_df.to_csv(rir_csv_path, index=False)
        print(f"   ✅ Created {rir_csv_path}")
        print(f"   📊 Unique RIR files: {len(rir_files)}")
        print(f"   📊 Total RIR entries: {len(rir_files_repeated)}")
    else:
        print(f"   ⚠️  RIR directory not found: {rir_dir}")
        print(f"   Please run generate_rir.py first!")
    
    # 4. 顯示示例
    print("\n" + "=" * 60)
    print("Sample entries:")
    print("=" * 60)
    if len(clean_files) > 0:
        print(f"\nClean: {clean_files[0]}")
    if 'noise_files_repeated' in locals() and len(noise_files_repeated) > 0:
        print(f"Noise: {noise_files_repeated[0]}")
    if 'rir_files_repeated' in locals() and len(rir_files_repeated) > 0:
        print(f"RIR:   {rir_files_repeated[0]}")
    
    print("\n" + "=" * 60)
    print("✅ All CSV lists generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    generate_csv_lists()
