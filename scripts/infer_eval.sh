#!/bin/bash
dataset_name="${1:-videomatte_1920x1080}"
sh scripts/inference_dataset.sh $dataset_name 
sh scripts/eval.sh $dataset_name