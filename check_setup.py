#!/usr/bin/env python3
"""
訓練前檢查腳本
確保所有必要的文件和配置都已就緒
"""
import os
from pathlib import Path
import sys

def check_item(condition, message, required=True):
    """檢查項目並打印結果"""
    if condition:
        print(f"  ✅ {message}")
        return True
    else:
        symbol = "❌" if required else "⚠️ "
        print(f"  {symbol} {message}")
        return not required

def main():
    print("=" * 60)
    print("GTCRN 訓練前檢查")
    print("=" * 60)
    
    all_ok = True
    
    # 1. 檢查數據文件
    print("\n1. 數據文件:")
    all_ok &= check_item(
        Path('datasets/train_data/train_noisy').exists(),
        "訓練集 noisy 目錄存在"
    )
    all_ok &= check_item(
        Path('datasets/train_data/train_clean').exists(),
        "訓練集 clean 目錄存在"
    )
    all_ok &= check_item(
        Path('datasets/val_data/train_noisy').exists(),
        "驗證集 noisy 目錄存在"
    )
    all_ok &= check_item(
        Path('datasets/val_data/train_clean').exists(),
        "驗證集 clean 目錄存在"
    )
    
    # 統計文件數量
    if Path('datasets/train_data/train_noisy').exists():
        train_count = len(list(Path('datasets/train_data/train_noisy').glob('*.wav')))
        print(f"     訓練樣本數: {train_count}")
    
    if Path('datasets/val_data/train_noisy').exists():
        val_count = len(list(Path('datasets/val_data/train_noisy').glob('*.wav')))
        print(f"     驗證樣本數: {val_count}")
    
    # 2. 檢查配置文件
    print("\n2. 配置文件:")
    all_ok &= check_item(
        Path('configs/cfg_train_custom.yaml').exists(),
        "訓練配置文件存在"
    )
    all_ok &= check_item(
        Path('dataloader_custom.py').exists(),
        "自定義 dataloader 存在"
    )
    all_ok &= check_item(
        Path('models/gtcrn_end2end.py').exists(),
        "模型文件存在"
    )
    all_ok &= check_item(
        Path('train.py').exists(),
        "訓練腳本存在"
    )
    
    # 3. 檢查輸出目錄
    print("\n3. 輸出目錄:")
    exp_dir = Path('experiments')
    check_item(
        exp_dir.exists(),
        f"實驗輸出目錄: {exp_dir}",
        required=False
    )
    if not exp_dir.exists():
        print("     (訓練時會自動創建)")
    
    # 4. 檢查 Python 依賴
    print("\n4. Python 依賴:")
    try:
        import torch
        check_item(True, f"PyTorch {torch.__version__}")
        
        # 檢查 CUDA
        if torch.cuda.is_available():
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("     ⚠️  CUDA 不可用，將使用 CPU（訓練會很慢）")
    except ImportError:
        all_ok &= check_item(False, "PyTorch")
    
    try:
        import soundfile
        check_item(True, "soundfile")
    except ImportError:
        all_ok &= check_item(False, "soundfile")
    
    try:
        import omegaconf
        check_item(True, "omegaconf")
    except ImportError:
        all_ok &= check_item(False, "omegaconf")
    
    # 5. 檢查 train.py 是否需要修改
    print("\n5. 訓練腳本配置:")
    with open('train.py', 'r') as f:
        train_content = f.read()
        if 'dataloader_custom' in train_content:
            check_item(True, "train.py 已更新為使用 dataloader_custom")
        else:
            all_ok &= check_item(
                False, 
                "train.py 需要修改第 21 行導入為 'from dataloader_custom import CustomDataset as Dataset'"
            )
    
    # 總結
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 所有檢查通過！可以開始訓練了")
        print("\n啟動訓練:")
        print("  bash start_training.sh")
        print("或:")
        print("  python train.py -C configs/cfg_train_custom.yaml -D 0")
    else:
        print("❌ 有些項目需要修復")
        print("\n請查看上面的錯誤訊息並修復後再試")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
