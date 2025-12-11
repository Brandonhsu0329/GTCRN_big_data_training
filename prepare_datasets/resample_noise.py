#!/usr/bin/env python3
"""
重採樣 NOISEX-92 噪音檔案從 19.98 kHz 到 16 kHz
"""
import os
import librosa
import soundfile as sf
from tqdm import tqdm

def resample_noise_files(input_dir, output_dir, target_sr=16000):
    """
    重採樣噪音檔案
    
    Args:
        input_dir: 輸入噪音目錄
        output_dir: 輸出目錄
        target_sr: 目標採樣率
    """
    os.makedirs(output_dir, exist_ok=True)
    
    noise_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    print(f"Found {len(noise_files)} noise files in {input_dir}")
    print(f"Resampling to {target_sr} Hz...")
    
    for noise_file in tqdm(noise_files):
        input_path = os.path.join(input_dir, noise_file)
        output_path = os.path.join(output_dir, noise_file)
        
        try:
            # 載入並重採樣
            y, sr = librosa.load(input_path, sr=target_sr)
            
            # 儲存
            sf.write(output_path, y, target_sr)
            
            # 顯示資訊
            if noise_file == noise_files[0]:  # 只顯示第一個檔案的資訊
                print(f"\nSample info:")
                print(f"  Original SR: {librosa.get_samplerate(input_path)} Hz")
                print(f"  Target SR: {target_sr} Hz")
                print(f"  Duration: {len(y)/target_sr:.2f} seconds")
                
        except Exception as e:
            print(f"\nError processing {noise_file}: {e}")
    
    print(f"\n✅ Resampling completed! Files saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    # 路徑設定
    noise_dir = '../speech_lib/noisex92'
    output_dir = '../speech_lib/noisex92_16k'
    
    # 執行重採樣
    resample_noise_files(noise_dir, output_dir, target_sr=16000)
