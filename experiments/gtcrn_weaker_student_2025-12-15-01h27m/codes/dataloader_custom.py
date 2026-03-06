from random import random
import soundfile as sf
import librosa
import torch
from torch.utils import data
import numpy as np
import random

# 更新為新的數據路徑
NOISY_DATABASE_TRAIN = 'datasets/train_data/train_noisy'
NOISY_DATABASE_VALID = 'datasets/val_data/train_noisy'

class CustomDataset(torch.utils.data.Dataset):
    """
    自定義數據集
    支持固定長度的音訊（補零或裁切）
    """
    def __init__(
        self,
        fs=16000,
        length_in_seconds=2,
        num_data_tot=1728,
        num_data_per_epoch=1000,
        random_start_point=False,
        train=True
    ):
        if train:
            print(f"Loading training data from: {NOISY_DATABASE_TRAIN}")
        else:
            print(f"Loading validation data from: {NOISY_DATABASE_VALID}")
        
        self.noisy_database_train = sorted(librosa.util.find_files(NOISY_DATABASE_TRAIN, ext='wav'))[:num_data_tot]
        self.noisy_database_valid = sorted(librosa.util.find_files(NOISY_DATABASE_VALID, ext='wav'))
        
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        
        print(f"  Sample length: {length_in_seconds} seconds ({self.L} samples)")
        print(f"  Total files: {len(self.noisy_database_train if train else self.noisy_database_valid)}")
        if train:
            print(f"  Samples per epoch: {num_data_per_epoch}")
        
    def sample_data_per_epoch(self):
        """每個 epoch 隨機抽取訓練樣本"""
        self.noisy_data_train = random.sample(self.noisy_database_train, 
                                             min(self.num_data_per_epoch, len(self.noisy_database_train)))

    def pad_or_truncate(self, audio, target_length):
        """補零或裁切音訊到目標長度"""
        current_length = len(audio)
        
        if current_length < target_length:
            # 補零
            padding = target_length - current_length
            audio = np.pad(audio, (0, padding), mode='constant')
        elif current_length > target_length:
            # 如果啟用隨機起始點，隨機裁切；否則從頭開始
            if self.random_start_point and current_length > target_length:
                start = np.random.randint(0, current_length - target_length + 1)
            else:
                start = 0
            audio = audio[start:start + target_length]
        
        return audio

    def __getitem__(self, idx):
        if self.train:
            noisy_list = self.noisy_data_train
        else:
            noisy_list = self.noisy_database_valid

        # 讀取 noisy 和 clean
        noisy, _ = sf.read(noisy_list[idx], dtype='float32')
        clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32')
        
        # 確保長度一致
        noisy = self.pad_or_truncate(noisy, self.L)
        clean = self.pad_or_truncate(clean, self.L)

        return noisy, clean

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.noisy_database_valid)


if __name__=='__main__':
    from tqdm import tqdm 
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train.yaml')

    train_dataset = CustomDataset(**config['train_dataset'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    train_dataloader.dataset.sample_data_per_epoch()

    validation_dataset = CustomDataset(**config['validation_dataset'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(f"\nTrain batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(validation_dataloader)}")

    print("\nTesting training dataloader...")
    for noisy, clean in tqdm(train_dataloader):
        print(f"Noisy shape: {noisy.shape}, Clean shape: {clean.shape}")
        break

    print("\nTesting validation dataloader...")
    for noisy, clean in tqdm(validation_dataloader):
        print(f"Noisy shape: {noisy.shape}, Clean shape: {clean.shape}")
        break
    
    print("\n✅ Dataloader test passed!")
