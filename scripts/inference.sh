#!/bin/bash

# 定义输入和输出的根目录
dataset_name="${1:-videomatte_1920x1080}"
INPUT_DIR="/home/xcai/caixin/RobustVideoMatting/data/$dataset_name"
OUTPUT_DIR="/home/xcai/caixin/RobustVideoMatting/data/pred/$dataset_name"
# 检索所有包含序列的子文件夹
for sequence_dir in $(ls $INPUT_DIR); do
    # 检查子目录是否存在
    if [ -d "$INPUT_DIR/$sequence_dir" ]; then
        # 遍历每个序列文件夹
        for sequence in $(ls $INPUT_DIR/$sequence_dir); do
            # 构建完整的输入输出路径
            input_path="$INPUT_DIR/$sequence_dir/$sequence/com"
            output_composition_path="$OUTPUT_DIR/$sequence_dir/$sequence/com"
            output_alpha_path="$OUTPUT_DIR/$sequence_dir/$sequence/pha"
            output_foreground_path="$OUTPUT_DIR/$sequence_dir/$sequence/fgr"

            # 运行 Python 脚本处理视频序列
            python inference.py \
                --variant mobilenetv3 \
                --checkpoint rvm_mobilenetv3.pth \
                --device cuda \
                --input-source $input_path \
                --output-type png_sequence \
                --output-composition $output_composition_path \
                --output-alpha $output_alpha_path \
                --output-foreground $output_foreground_path \
                --output-video-mbps 4 \
                --num-workers 8 \
                --seq-chunk 1
        done
    fi
done