"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm

from inference import Converter
    

def find_directories(root_path, dir_name):
    matches = []
    for root, dirs, files in os.walk(root_path):
        for directory in dirs:
            if directory == dir_name:
                matches.append(os.path.join(root, directory))
    return matches

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="dataset name")
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument("--comp-dir", default="com", type=str, help="composition directory")
    parser.add_argument("--dataset_root", type=str, default="data/evaluation")
    parser.add_argument("--output_root", type=str, default="data/pred")
    parser.add_argument('--disable-progress', action='store_true')
    args = parser.parse_args()
    converter = Converter(args.variant, args.checkpoint, args.device)
    
    dataset_root = os.path.join(args.dataset_root, args.dataset)
    print(dataset_root)
    output_root = os.path.join(args.output_root, args.dataset)
    all_compositions = find_directories(dataset_root, args.comp_dir)
    for composition in all_compositions:
        composition_name = composition.split(dataset_root)[-1].split(args.comp_dir)[-2].strip('/')
        print(f"Processing {composition_name}")
        output_fgr_path = os.path.join(output_root, composition_name, "fgr")
        output_pha_path = os.path.join(output_root, composition_name, "pha")
        output_com_path = os.path.join(output_root, composition_name, "com")   
       
        converter.convert(
            input_source=composition,
            input_resize=args.input_resize,
            downsample_ratio=args.downsample_ratio,
            output_type=args.output_type,
            output_composition=output_com_path,
            output_alpha=output_pha_path,
            output_foreground=output_fgr_path,
            output_video_mbps=args.output_video_mbps,
            seq_chunk=args.seq_chunk,
            num_workers=args.num_workers,
            progress=not args.disable_progress
        )
    
    
