# simluation single burst for visualization purpose

import argparse
import os
import pims
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as F
import sys
sys.path.append('.')
from utils.image.synthesis_helper import get_unprocessing_settings,unprocess_images, apply_noise
import gin
 
def rgb2raw(image, randomize_gain=False, add_noise=False):
    """Convert an RGB image to a 'raw' image simulating camera processing."""
    unprocessing_params = get_unprocessing_settings()
    image, unprocess_metadata = unprocess_images(image, unprocessing_params, randomize_gain)
    # Apply noise if specified
    if add_noise:
        image = apply_noise(image, unprocessing_params)
    return image.clamp(0.0, 1.0), unprocess_metadata

def lerp(a, b, percentage):
    return a * (1 - percentage) + b * percentage

def motion_affine(*imgs):
    config = dict(degrees=(-10, 10), translate=(0.1, 0.1),
                  scale_ranges=(0.9, 1.1), shears=(-5, 5), img_size=imgs[0][0].size)
    angleA, (transXA, transYA), scaleA, (shearXA, shearYA) = transforms.RandomAffine.get_params(**config)
    angleB, (transXB, transYB), scaleB, (shearXB, shearYB) = transforms.RandomAffine.get_params(**config)

    T = len(imgs[0])
    variation_over_time = random.random()
    for t in range(T):
        percentage = (t / (T - 1)) * variation_over_time
        angle = lerp(angleA, angleB, percentage)
        transX = lerp(transXA, transXB, percentage)
        transY = lerp(transYA, transYB, percentage)
        scale = lerp(scaleA, scaleB, percentage)
        shearX = lerp(shearXA, shearXB, percentage)
        shearY = lerp(shearYA, shearYB, percentage)
        for img in imgs:
            img[t] = F.affine(img[t], angle, (transX, transY), scale, (shearX, shearY), F.InterpolationMode.BILINEAR)
    return imgs

def process(bgr_path, fgr_path, pha_path, out_path, num_frames = 20, resolution = 1024):
    os.makedirs(os.path.join(out_path, 'fgr'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'pha'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'com'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'bgr'), exist_ok=True)
    extension = '.png'
    with Image.open(os.path.join(bgr_path)) as bgr:
        bgr = bgr.convert('RGB')
        
        w, h = bgr.size
        scale = resolution / min(h, w)
        w, h = int(w * scale), int(h * scale)
        bgr = bgr.resize((w, h))
        bgr = F.center_crop(bgr, (resolution, resolution))

    with Image.open(os.path.join(fgr_path)) as fgr, \
         Image.open(os.path.join(pha_path)) as pha:
        fgr = fgr.convert('RGB')
        pha = pha.convert('L')
    bgr = bgr.transpose(Image.FLIP_LEFT_RIGHT)
    fgrs = [fgr] * num_frames
    phas = [pha] * num_frames
    bgrs = [bgr] * num_frames
    fgrs, phas = motion_affine(fgrs, phas)
    bgrs = motion_affine(bgrs)[0]

    for t in tqdm(range(num_frames)):
        fgr = fgrs[t]
        pha = phas[t]
        bgr = bgrs[t]
        w, h = fgr.size
        scale = resolution / max(h, w)
        w, h = int(w * scale), int(h * scale)
        
        fgr = fgr.resize((w, h))
        pha = pha.resize((w, h))
        
        if h < resolution:
            pt = (resolution - h) // 2
            pb = resolution - h - pt
        else:
            pt = 0
            pb = 0
            
        if w < resolution:
            pl = (resolution - w) // 2
            pr = resolution - w - pl
        else:
            pl = 0
            pr = 0
            
        fgr = F.pad(fgr, [pl, pt, pr, pb])
        pha = F.pad(pha, [pl, pt, pr, pb])
        

            
        fgr.save(os.path.join(out_path, 'fgr', str(t).zfill(4) + extension))
        pha.save(os.path.join(out_path, 'pha', str(t).zfill(4) + extension))
        bgr.save(os.path.join(out_path, 'bgr', str(t).zfill(4) + extension))
        
        # if t == 0:
        #     bgr.save(os.path.join(out_path, 'bgr', str(t).zfill(4) + extension))
        # else:
        #     os.symlink(str(0).zfill(4) + extension, os.path.join(out_path, 'bgr', str(t).zfill(4) + extension))
        
        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_path, 'com', str(t).zfill(4) + extension))
    # save the composite image to gif
    com_gif = []
    for t in range(num_frames):
        com_gif.append(Image.open(os.path.join(out_path, 'com', str(t).zfill(4) + extension)))
    com_gif[0].save(os.path.join(out_path, 'com.gif'), save_all=True, append_images=com_gif[1:], loop=0, duration=100)



if __name__ == '__main__':
    bgr_path = '/home/xcai/caixin/RobustVideoMatting/image.png'
    fgr_path = '/home/xcai/caixin/RobustVideoMatting/data/matting-dataset/Combined_Dataset/Training_set/Adobe-licensed images/fg/1-1259245823Un3j.jpg'
    pha_path = '/home/xcai/caixin/RobustVideoMatting/data/matting-dataset/Combined_Dataset/Training_set/Adobe-licensed images/alpha/1-1259245823Un3j.jpg'
    out_path = 'data/sim_single_burst/1-1259245823Un3j'
    # if os.path.exists(out_path):
    #     os.system('rm -r ' + out_path)
    # process(bgr_path, fgr_path, pha_path, out_path)

    gin.parse_config_file('configs/matting/image2burst.gin')

    raw_gif = []
    num_frames = 20
    extension = '.png'
    # unprocess the composite images to raw images
    for t in range(num_frames):
        com = Image.open(os.path.join(out_path, 'com', str(t).zfill(4) + extension))
        com_tensor = transforms.ToTensor()(com).unsqueeze(0)
        raw, _ = rgb2raw(com_tensor)
        raw = transforms.ToPILImage()(raw.squeeze(0))
        raw_gif.append(raw)
    raw_gif[0].save(os.path.join(out_path, 'raw.gif'), save_all=True, append_images=raw_gif[1:], loop=0, duration=100)