# generate test burst data for testing, this is a script to generate test data for testing
import sys
sys.path.append('.')
sys.path.append('..')
from dataset.synthetic_burst import SyntheticBurstDataset
import os
import gin
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from utils.image.synthesis_helper import get_motion_aug_params
from tqdm import tqdm
from dataset.augmentation import MotionAugmentation

gin.parse_config_file('configs/tools/gen_test_burst.gin')


motion_aug_params = get_motion_aug_params()
motion_aug = MotionAugmentation(**motion_aug_params)
matting_subset = ([0,1], 2)
split = "test"
output_dir = "data/burst_aim_test"
dataset = SyntheticBurstDataset(matting_subset = matting_subset, split = split, motion_aug = motion_aug)
for i, data in  tqdm(enumerate(dataset)):
    output_dir_for_data = os.path.join(output_dir, "%04d" % i)
    os.makedirs(output_dir_for_data, exist_ok=True)
    alpha_burst = data['alpha_burst']
    fgrs_gamma = data['fgrs_gamma']
    bgrs_gamma = data['bgrs_gamma']
    comp_burst_gamma = data['comp_burst_gamma']
    os.makedirs(os.path.join(output_dir_for_data, "pha"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_for_data, "fgr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_for_data, "bgr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_for_data, "com"), exist_ok=True)
    for j in range(len(alpha_burst)):
        vutils.save_image(alpha_burst[j], os.path.join(output_dir_for_data, f"pha/{j:04}.png"))
        vutils.save_image(fgrs_gamma[j], os.path.join(output_dir_for_data, f"fgr/{j:04}.png"))
        vutils.save_image(bgrs_gamma[j], os.path.join(output_dir_for_data, f"bgr/{j:04}.png"))
        vutils.save_image(comp_burst_gamma[j], os.path.join(output_dir_for_data, f"com/{j:04}.png"))
