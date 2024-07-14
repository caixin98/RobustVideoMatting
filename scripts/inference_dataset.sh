#!/bin/bash
dataset_name="${1:-videomatte_1920x1080}"
python inference_dataset.py $dataset_name \
    --variant mobilenetv3 \
    --checkpoint rvm_mobilenetv3.pth \
    --device cuda \
    --output-type png_sequence \
    --output-video-mbps 4 \
    --num-workers 8 \
    --seq-chunk 1 \
    --disable-progress 