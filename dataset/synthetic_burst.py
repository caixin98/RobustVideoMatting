import os
import random
from torch.utils.data import Dataset
from PIL import Image
from dataset.fgr_phas import ForegroundAlphaDataset
from dataset.bg_imgs import BackgroundImageDataset, ZurichRGBDataset
from torch.utils.data import ConcatDataset
# from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip
import torch.nn.functional as F
import torch
from .augmentation import MotionAugmentation
from utils.image.synthesis_helper import generate_flow_field, apply_flow_field
from utils.image.synthesis_helper import get_unprocessing_settings,unprocess_images, apply_noise
from utils.image.trimap import make_trimap
from utils.image.rgb2raw import process_linear_image_raw

class ImageMatteAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=1.0,
            prob_bgr_affine=0.0,
            prob_noise=0.05,
            prob_color_jitter=0.3,
            prob_grayscale=0.03,
            prob_sharpness=0.05,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
            static_affine=False,
            cuda=True,
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
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.split = split
        self.matt_dataset = ForegroundAlphaDataset(matting_subset, split=split)
        bg_dataset_rgb =  BackgroundImageDataset(bg_subset, split=split)
        # bg_dataset_raw = ZurichRGBDataset(split=split)
        # self.bg_dataset = ConcatDataset([bg_dataset_rgb, bg_dataset_raw])
        self.bg_dataset = bg_dataset_rgb
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
 
    def rgb2raw(self, image, randomize_gain=True, add_noise=True, device='cuda'):
        """Convert an RGB image to a 'raw' image simulating camera processing."""
        # unprocessing_params = get_unprocessing_settings()

        unprocessing_params = {"random_rgb_gain_range":[0.9, 1.1],
                                 "random_ccm":False,
                                 "random_gains":True,
                                 "smoothstep":True,
                                 "gamma":True,
                                 "add_noise":False,
                                 "noise_type":'samsung'}
        image = image.to(device)    
        image, unprocess_metadata = unprocess_images(image, unprocessing_params, randomize_gain)
        # Apply noise if specified
        if add_noise:
            image = apply_noise(image, unprocessing_params)
        return image.clamp(0.0, 1.0), unprocess_metadata



        # fgr_linear = fgrs_linear[0]
        # bgr_linear = bgrs_linear[0]
        # fgr_gamma = fgrs_gamma[0]
        # bgr_gamma = bgrs_gamma[0]
        # alpha = alpha_burst[0]
        # comp_gamma = fgr_gamma * alpha + bgr_gamma * (1 - alpha)
        # comp_linear = fgr_linear * alpha + bgr_linear * (1 - alpha)
        # comp_burst_gamma = fgrs_gamma * alpha_burst + bgrs_gamma * (1 - alpha_burst)

    def __getitem__(self, idx):
        import time 
        fgr_gamma, phas = self._get_fgr_phas(idx) # get the foreground and alpha, fgr_gamma in the srgb color space (gamma space)
        bgr_gamma = self._get_random_image_background()
        fgrs_gamma, alpha_burst, bgrs_gamma, motion_metadata = self.data2burst(fgr_gamma, phas, bgr_gamma)
        fgrs_bgrs_gamma = torch.cat([fgrs_gamma,bgrs_gamma], dim=0)
        # convert the gamma space to linear space
        fgrs_bgrs_linear, unprocess_metadata = self.rgb2raw(fgrs_bgrs_gamma, device='cuda')
        trimap_burst, alpha_burst = make_trimap(alpha_burst, dilation_size=20, apply_closing=True, device='cuda')
        fgrs_linear = fgrs_bgrs_linear[:self.burst_length]
        bgrs_linear = fgrs_bgrs_linear[self.burst_length:]
        comp_burst = fgrs_linear * alpha_burst + bgrs_linear * (1 - alpha_burst)

        unprocess_metadata["cam2rgb"] = unprocess_metadata["cam2rgb"][0]
        # comp_burst_gamma = process_linear_image_raw(comp_burst[0:1], unprocess_metadata, apply_demosaic=False)
        datapoint = {'unprocess_metadata': unprocess_metadata, "motion_metadata": motion_metadata}

        # for k in ("alpha", 'fgr_gamma', 'bgr_gamma', 'fgr_linear', 'bgr_linear', 'comp_gamma','comp_linear', 
                #   'fgrs_linear', 'bgrs_linear', 'fgrs_gamma', 'bgrs_gamma', 'alpha_burst', 'comp_burst', 'comp_burst_gamma', 'trimap_burst'):

        for k in ('comp_burst', 'trimap_burst',  "fgrs_linear", 'bgrs_linear', 'alpha_burst'):
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
        if new_width >= target_size_width and new_height >= target_size_height:
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

# output the burst image patches that trimap are not totally background or foreground
class SyntheticBurstPatch(SyntheticBurstDataset):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.current_group = []
        self.patch_ratio = 0.05
        self.group_idx = 0
    
    def split_into_patches(self, data, patch_size, padding_mode='constant'):
        # Calculate required padding
        _, _, height, width = data.shape
        pad_height = (patch_size[0] - height % patch_size[0]) % patch_size[0]
        pad_width = (patch_size[1] - width % patch_size[1]) % patch_size[1]

        # Pad the data on the bottom and right edges
        data = F.pad(data, (0, pad_width, 0, pad_height), mode=padding_mode)

        # Now use unfold
        patches = data.unfold(2, patch_size[0], patch_size[0]).unfold(3, patch_size[1], patch_size[1])

        # patches shape: [batch_size, channels, height_patches, width_patches, patch_height, patch_width]
        # Reshape and permute to bring the patch indices to the first dimension and keep batch size separate
        batch_size, channels, height_patches, width_patches, patch_height, patch_width = patches.shape
        patches = patches.permute(2, 3, 0, 1, 4, 5).contiguous().view(-1, batch_size, channels, patch_height, patch_width)
        # patches = patches.contiguous().view(-1, data.size(1), patch_size[0], patch_size[1])
        return patches

    def load_one_burst(self):
        # random select one burst from the dataset
        idx = random.randint(0,  super().__len__() - 1)
        data = super().__getitem__(idx)
        self.group_idx += 1
        patch_data = {}
        for k in ('fgrs_linear', 'bgrs_linear', 'alpha_burst', 'comp_burst', 'trimap_burst'):
            if k in data:
                patch_data[k] = self.split_into_patches(data[k], self.patch_size, padding_mode='constant')
        # Get the trimap patches
        trimap_patches = patch_data['trimap_burst']
        sum_list = []
        for i in range(trimap_patches.size(0)):
            sum_list.append(torch.sum(trimap_patches[i,:,1]))
        # according the trimap to get the top 20% patches with unknown region
        top_k = int(self.patch_ratio * len(sum_list))
        top_k_idx = sorted(range(len(sum_list)), key=lambda i: sum_list[i])[-top_k:]
        for k in ('fgrs_linear', 'bgrs_linear', 'alpha_burst', 'comp_burst', 'trimap_burst'):
            if k in patch_data:
                patch_data[k] = patch_data[k][top_k_idx]
    
        patch_data_group = []
        for i in range(top_k):
            patch_data_case = {k: v[i] for k, v in patch_data.items()}
            patch_data_case['unprocess_metadata'] = data['unprocess_metadata']
            patch_data_group.append(patch_data_case)
        return patch_data_group

    

    # def __getitem__(self, idx):
    #     data = super().__getitem__(idx)
    #     for k in ('fgrs_linear', 'bgrs_linear', 'alpha_burst', 'comp_burst', 'trimap_burst'):
    #         if k in data:
    #             data[k] = self.split_into_patches(data[k], self.patch_size, padding_mode='constant')
    #     return data

    def __getitem__(self, idx):
        if len(self.current_group) == 0:
            self.current_group = self.load_one_burst()
        data = self.current_group.pop()
        return data 

    def __len__(self):
        width, height = self.size
        patch_width, patch_height = self.patch_size
        num_patches_width = width // patch_width + (width % patch_width > 0)
        num_patches_height = height // patch_height + (height % patch_height > 0)
        group_size = int(num_patches_height * num_patches_width * self.patch_ratio)
        # return int(super().__len__() * group_size)
        if self.split == 'train':
            return 2000
        else:
            return 500


        
def reconstruct_from_patches(patches, original_shape):
    # Assuming original_shape is in the format (batch_size, channels, height, width)
    # Patches are assumed to be non-overlapping
    # Create an empty tensor for the reconstructed data
    reconstructed = torch.zeros(original_shape)
    # Get patch size
    patch_height, patch_width = patches.size(3), patches.size(4)

    # Number of patches along height and width
    num_patches_height = original_shape[2] // patch_height
    num_patches_width = original_shape[3] // patch_width

    patch_idx = 0
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            reconstructed[:, :, i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width] = patches[patch_idx, :, :, :, :]
            patch_idx += 1
            
    return reconstructed