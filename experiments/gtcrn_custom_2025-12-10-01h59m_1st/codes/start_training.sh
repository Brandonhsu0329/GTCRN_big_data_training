#!/bin/bash
# GTCRN 訓練啟動腳本

echo "======================================================"
echo "  GTCRN Training Script"
echo "======================================================"
echo ""

# 檢查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No GPU detected, will use CPU (very slow!)"
fi

echo ""
echo "Configuration:"
echo "  - Audio length: 2 seconds"
echo "  - Batch size: 16"
echo "  - Epochs: 100"
echo "  - Training samples: 1728"
echo "  - Validation samples: 192"
echo ""

# 詢問是否繼續
read -p "Start training? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting training..."
    echo "Output will be saved to: experiments/gtcrn_custom/"
    echo ""
    echo "Monitor with TensorBoard:"
    echo "  tensorboard --logdir experiments/gtcrn_custom/logs"
    echo ""
    
    # 啟動訓練（需要先修改 train.py 的導入）
    python train.py -C configs/cfg_train_custom.yaml -D 0
else
    echo "Training cancelled."
fi
