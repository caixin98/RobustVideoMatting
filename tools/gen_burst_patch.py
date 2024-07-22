# generate burst data patch with SyntheticBurstPatch dataset, this is a script to generate data for training and testing 
import sys
sys.path.append('.')
sys.path.append('..')
from dataset.synthetic_burst import SyntheticBurstPatch
import os
import gin
from dataset.augmentation import MotionAugmentation
from utils.image.synthesis_helper import get_motion_aug_params
from tqdm import tqdm
import torchvision.utils as vutils

gin.parse_config_file('configs/tools/gen_burst_patch.gin')

motion_aug_params = get_motion_aug_params()

motion_aug = MotionAugmentation(**motion_aug_params)

matting_subset = ([3], 4)
split = "test"
output_dir = "data/burst_patch_evaluation"
dataset = SyntheticBurstPatch(patch_size=192, size=1024, matting_subset = matting_subset, split = split, motion_aug = motion_aug)
unprocess_metadata_json = os.path.join(output_dir, "unprocess_metadata.json")
unprocess_metadatas = []
for i, data in  tqdm(enumerate(dataset)):
    output_dir_for_data = os.path.join(output_dir, "%04d" % i)
    os.makedirs(output_dir_for_data, exist_ok=True)
    alpha_burst = data['alpha_burst']
    fgrs_linear = data['fgrs_linear']
    bgrs_linear = data['bgrs_linear']
    comp_burst = data['comp_burst']
    trimap_burst = data['trimap_burst']
    unprocess_metadata = data['unprocess_metadata']
    unprocess_metadatas.append({"output_dir": output_dir_for_data, "unprocess_metadata": unprocess_metadata})
    os.makedirs(os.path.join(output_dir_for_data, "pha"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_for_data, "fgr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_for_data, "bgr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_for_data, "com"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_for_data, "tri"), exist_ok=True)
    for j in range(len(alpha_burst)):
        vutils.save_image(alpha_burst[j], os.path.join(output_dir_for_data, f"pha/{j:04}.png"))
        vutils.save_image(fgrs_linear[j], os.path.join(output_dir_for_data, f"fgr/{j:04}.png"))
        vutils.save_image(bgrs_linear[j], os.path.join(output_dir_for_data, f"bgr/{j:04}.png"))
        vutils.save_image(comp_burst[j], os.path.join(output_dir_for_data, f"com/{j:04}.png"))
        vutils.save_image(trimap_burst[j], os.path.join(output_dir_for_data, f"tri/{j:04}.png"))
        
import json
with open(unprocess_metadata_json, 'w') as f:
    json.dump(unprocess_metadatas, f)
print(f"Saved unprocess metadata to {unprocess_metadata_json}")