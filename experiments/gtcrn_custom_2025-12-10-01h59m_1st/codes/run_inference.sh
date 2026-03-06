#!/bin/bash
# GTCRN 推論範例腳本

echo "======================================================"
echo "  GTCRN Inference Examples"
echo "======================================================"
echo ""

# 設置路徑（根據你的實際情況修改）
CHECKPOINT="experiments/gtcrn_custom_2025-12-10-00h28m/checkpoints/best_model_001.tar"
DEVICE="cuda"  # 或 "cpu"

echo "使用的模型: $CHECKPOINT"
echo ""
echo "請選擇推論模式："
echo "1) 單個文件推論"
echo "2) 批次推論（處理整個目錄）"
echo "3) 測試驗證集樣本"
echo ""
read -p "請選擇 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "=== 單個文件推論 ==="
        read -p "輸入音訊路徑: " input_file
        read -p "輸出音訊路徑 (按 Enter 使用默認): " output_file
        
        if [ -z "$output_file" ]; then
            output_file="${input_file%.*}_enhanced.wav"
        fi
        
        echo ""
        echo "開始處理..."
        python inference.py \
            -c "$CHECKPOINT" \
            -d "$DEVICE" \
            -i "$input_file" \
            -o "$output_file"
        ;;
        
    2)
        echo ""
        echo "=== 批次推論 ==="
        read -p "輸入目錄路徑: " input_dir
        read -p "輸出目錄路徑 (按 Enter 使用默認): " output_dir
        
        if [ -z "$output_dir" ]; then
            output_dir="${input_dir}_enhanced"
        fi
        
        echo ""
        echo "開始批次處理..."
        python inference.py \
            -c "$CHECKPOINT" \
            -d "$DEVICE" \
            -i "$input_dir" \
            -o "$output_dir" \
            --batch
        ;;
        
    3)
        echo ""
        echo "=== 測試驗證集樣本 ==="
        
        # 使用驗證集的一個樣本
        VAL_DIR="datasets/val_data/train_noisy"
        OUTPUT_DIR="inference_results/val_samples"
        
        if [ ! -d "$VAL_DIR" ]; then
            echo "❌ 驗證集目錄不存在: $VAL_DIR"
            exit 1
        fi
        
        echo "輸入: $VAL_DIR"
        echo "輸出: $OUTPUT_DIR"
        echo ""
        
        python inference.py \
            -c "$CHECKPOINT" \
            -d "$DEVICE" \
            -i "$VAL_DIR" \
            -o "$OUTPUT_DIR" \
            --batch
        
        echo ""
        echo "你可以比較以下文件："
        echo "  原始 noisy: $VAL_DIR/"
        echo "  增強後: $OUTPUT_DIR/"
        echo "  真實 clean: datasets/val_data/train_clean/"
        ;;
        
    *)
        echo "無效的選擇"
        exit 1
        ;;
esac

echo ""
echo "======================================================"
echo "完成！"
echo "======================================================"
