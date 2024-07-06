import os
import random
from torch.utils.data import Dataset
from PIL import Image
from dataset.fgr_phas import ForegroundAlphaDataset
from dataset.bg_imgs import BackgroundImageDataset, ZurichRGBDataset
from torch.utils.data import ConcatDataset
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip
import torch.nn.functional as F
import torch
from .augmentation import MotionAugmentation
from utils.image.synthesis_helper import generate_flow_field, apply_flow_field
from utils.image.synthesis_helper import get_unprocessing_settings,unprocess_images, apply_noise



class ImageMatteAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.95,
            prob_bgr_affine=0.3,
            prob_noise=0.05,
            prob_color_jitter=0.3,
            prob_grayscale=0.03,
            prob_sharpness=0.05,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )


class SyntheticBurstDataset(Dataset):
    def __init__(self,
                size = (1920,1080), # the size of the image, (width, height)
                matting_subset = ([0], 1), # use which subset of matting dataset for training and testing
                bg_subset = ([0], 1), # use which subset of background dataset for training and testing
                split = 'train',
                burst_length = 8,
                motion_aug = None,
                seq_sampler = 'random'):
        self.size = size
        self.split = split
        self.matt_dataset = ForegroundAlphaDataset(matting_subset, split=split)
        bg_dataset_rgb =  BackgroundImageDataset(bg_subset, split=split)
        bg_dataset_raw = ZurichRGBDataset(split=split)
        self.bg_dataset = ConcatDataset([bg_dataset_rgb, bg_dataset_raw])
        # tranfer the size (width, height) to (height, width)
        size = (size[1], size[0])
        if motion_aug is None:
            self.motion_aug = ImageMatteAugmentation(size=size)
        else:
            self.motion_aug = motion_aug
        self.burst_length = burst_length



    def data2burst(self, fgr, phas, bgr): # augment the data to burst images with motion augmentation 
        fgrs = [fgr] * self.burst_length
        phas = [phas] * self.burst_length
        bgrs = [bgr] * self.burst_length
        fgrs, phas, bgrs, metadata = self.motion_aug(fgrs, phas, bgrs)
        fgrs_size = fgrs[0].size()[-2:]
        bgrs_size = bgrs[0].size()[-2:]
        fgr_flow_fields = []
        bgr_flow_fields = []
        for i in range(self.burst_length):
            fgr_flow_fields.append(generate_flow_field(metadata["final_affine_matrixs_fgr"][i], fgrs_size))
            bgr_flow_fields.append(generate_flow_field(metadata["final_affine_matrixs_bgr"][i], bgrs_size))
        metadata['fgr_flow_fields'] = fgr_flow_fields
        metadata['bgr_flow_fields'] = bgr_flow_fields
        return fgrs, phas, bgrs, metadata
 
    def rgb2raw(self, image, randomize_gain=True, add_noise=True):
        """Convert an RGB image to a 'raw' image simulating camera processing."""
        unprocessing_params = get_unprocessing_settings()
        image, unprocess_metadata = unprocess_images(image, unprocessing_params, randomize_gain)
        # Apply noise if specified
        if add_noise:
            image = apply_noise(image, unprocessing_params)
        return image.clamp(0.0, 1.0), unprocess_metadata




    def __getitem__(self, idx):
        fgr_gamma, phas = self._get_fgr_phas(idx) # get the foreground and alpha, fgr_gamma in the srgb color space (gamma space)
        # TODO:to change the img2burst function, make it only return a linear image
        # then we use motion augmentation to generate the burst images
        bgr_gamma = self._get_random_image_background()

        fgrs_gamma, alpha_burst, bgrs_gamma, motion_metadata = self.data2burst(fgr_gamma, phas, bgr_gamma)

        fgrs_bgrs_gamma = torch.cat([fgrs_gamma,bgrs_gamma], dim=0)
        # convert the gamma space to linear space
        fgrs_bgrs_linear, unprocess_metadata = self.rgb2raw(fgrs_bgrs_gamma)
        fgrs_linear = fgrs_bgrs_linear[:self.burst_length]
        bgrs_linear = fgrs_bgrs_linear[self.burst_length:]
        fgr_linear = fgrs_linear[0]
        bgr_linear = bgrs_linear[0]
        fgr_gamma = fgrs_gamma[0]
        bgr_gamma = bgrs_gamma[0]
        alpha = alpha_burst[0]
        comp_burst = fgrs_linear * alpha_burst + bgrs_linear * (1 - alpha_burst)
        comp_gamma = fgr_gamma * alpha + bgr_gamma * (1 - alpha)
        comp_linear = fgr_linear * alpha + bgr_linear * (1 - alpha)
        comp_burst_gamma = fgrs_gamma * alpha_burst + bgrs_gamma * (1 - alpha_burst)

        datapoint = {'unprocess_metadata': unprocess_metadata, "motion_metadata": motion_metadata}
        for k in ("alpha", 'fgr_gamma', 'bgr_gamma', 'fgr_linear', 'bgr_linear', 'comp_gamma',
                  'comp_linear', 'fgrs_linear', 'bgrs_linear',
                   'fgrs_gamma', 'bgrs_gamma', 'alpha_burst', 'comp_burst', 'comp_burst_gamma'):
            if k in locals():
                datapoint[k] = locals()[k]

        return datapoint
    
    def __len__(self):  
        return len(self.matt_dataset)
    
    
    def _get_fgr_phas(self, idx):
        data = self.matt_dataset[idx]
        fgr = data['image']
        phas = data['alpha']
        fgr, phas = self._random_crop_resize_and_padding_fgr_phas(fgr, phas)
        
        # fgr = self._downsample_if_needed(fgr)
        # phas = self._downsample_if_needed(phas)
        return fgr, phas
    
    def _get_random_image_background(self):
        data = random.choice(self.bg_dataset)
        # image = data['image']
        image = self._random_resize_and_crop_background(data['image'])
        return image

    def _downsample_if_needed(self, img):
        w, h = img.size
        scale = min(self.size[0] / w, self.size[1] / h, 1)
        print(w,h,self.size,scale)
        w = int(scale * w)
        h = int(scale * h)
        img = img.resize((w, h))
        return img

    def _random_resize_and_crop_background(self, image):
        target_size = self.size
        width, height = image.size
        target_size_width = target_size[0]
        target_size_height = target_size[1]
        # Calculate scaling factors to ensure the new size surrounds the given size and is enclosed by 2*size
        scale_min = max(target_size_width / width, target_size_height / height)
        scale_max = 1.5 * scale_min
        scale_factor = random.uniform(scale_min, scale_max)

        # scale_max = max((2 * target_size_width) / width, (2 * target_size_height) / height)

        # Randomly choose a scaling factor within the allowable range
        # if scale_min < scale_max:
            # scale_factor = random.uniform(scale_min, scale_max)
        # else:
        #     # If the minimum scaling factor is greater than or equal to the maximum, the parameters are incompatible
        #     raise ValueError("Given size is incompatible with the original dimensions of the image")
        
        # Calculate the new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize the image to the new dimensions
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # Calculate the starting point for the crop
        if new_width > target_size_width and new_height > target_size_height:
            x = random.randint(0, new_width - target_size_width)
            y = random.randint(0, new_height - target_size_height)
            cropped_image = resized_image.crop((x, y, x + target_size_width, y + target_size_height))
            return cropped_image
        else:
            # If resized dimensions are still not sufficient for cropping, raise an error
            raise ValueError("Resized dimensions are still not sufficient for the required crop size")

    def _random_crop_resize_and_padding_fgr_phas(self, fgr, phas, min_crop_ratio=0.8):
        # randomly crop the foreground with 0.8-1.0 ratio
        # resize the cropped image to the target size and pad the image if necessary

        crop_ratio = random.uniform(min_crop_ratio, 1.0)
        width, height = fgr.size
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)
        cropped_fgr = fgr.crop((x, y, x + crop_width, y + crop_height))
        cropped_phas = phas.crop((x, y, x + crop_width, y + crop_height))

        target_size = self.size
        padded_fgr = self._resize_and_padding_to_target_size(cropped_fgr, target_size)
        padded_phas = self._resize_and_padding_to_target_size(cropped_phas, target_size)
        return padded_fgr, padded_phas

    def _resize_and_padding_to_target_size(self, image, target_size):
        # resize the cropped image to the target size and pad the image if necessary

        width, height = image.size
        target_size_width = target_size[0]
        target_size_height = target_size[1]
        scale_factor = min(target_size_width / width, target_size_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)
        padding_left = int((target_size_width - new_width) / 2)
        padding_top = int((target_size_height - new_height) / 2)
        padding_right = target_size_width - new_width - padding_left
        padding_bottom = target_size_height - new_height - padding_top
        padded_image = Image.new(image.mode, (target_size_width, target_size_height), 0)
        padded_image.paste(resized_image, (padding_left, padding_top))
        return padded_image


    # def __len__(self):
    #     return len(self.burst_files)
    
    # def __getitem__(self, idx):
    #     with Image.open(os.path.join(self.burst_dir, self.burst_files[idx])) as burst:
    #         burst = burst.convert('RGB')
        
    #     if self.transform is not None:
    #         burst = self.transform(burst)
        
    #     return burst