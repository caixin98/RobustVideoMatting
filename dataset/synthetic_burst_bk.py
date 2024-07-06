"""
Extends Pytorch torch.utils.data.Dataset and LightningDataModule for synthetic composite bursts.
Configured to use Adobe Image Matting, polarized masks and video green screen
datasets for foreground, and the Zurich dataset for background.
"""
import pdb
import gin
import pytorch_lightning as pl
import torch

F = torch.nn.functional
from .burst.matting import MattingDataset, VideoMattingDataset
from .burst.zurich import ZurichRGBDataset, prepare_zurich_data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, PILToTensor, Compose, RandomHorizontalFlip, CenterCrop

from utils.image.image2burst import GenerateBurst


@gin.configurable
class SynthMattingDataModule(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 num_workers,
                 fg_source,
                 bg_source,
                 samples_per_epoch=None) -> None:
        super().__init__()
        assert fg_source in ['matting', 'polarized', 'video_gs']
        assert bg_source in ['zurich']
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        prepare_zurich_data()

    def train_dataloader(self):
        dataset = SyntheticMattingDataset(
            samples_per_epoch=self.samples_per_epoch,
            split='train',
            shuffle=True)
        return DataLoader(dataset,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        dataset = SyntheticMattingDataset(split='test', shuffle=False)
        return DataLoader(dataset,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          shuffle=False)


@gin.configurable
class SyntheticMattingDataset(torch.utils.data.Dataset):
    """Produces a linear RGB burst from unprocessing + simulated motion on single
    foreground and background RGB images. Burst length controlled by GenerateBurst().

    Inputs:
        bg_movement: if False, background image does not move during burst (noise is still added)
        num_fgs: if not None, only use this many foreground images
        num_bgs: if not None, only use this many background images
        border_crop: crop the burst so that padding is not needed when adding motion

    Output is a dict with the following keys:
        fg_srgb: unmodified foreground image (single frame)
        bg_srgb: unmodified background image (single frame)
        fg_lrgb: unprocessed version of fg_srgb (no noise or motion added)
        bg_lrgb: unprocessed version of bg_lrgb (no noise or motion added)
        fg_burst: add noise and motion to fg_lrgb (motion not applied to first frame)
        bg_burst: add noise and motion to bg_lrgb (motion not applied to first frame)
        alpha_burst: alpha matte of foreground for every frame
        comp_srgb: alpha_burst * fg_srgb + (1-alpha_burst) * bg_srgb
        comp_lrgb: alpha_burst * fg_lrgb + (1-alpha_burst) * bg_lrgb
        comp_burst: alpha_burst * fg_burst + (1-alpha_burst) * bg_burst
        fg_path: path to original foreground image
        metadata: dict with following keys:
            shifts: motion shifts applied to each frame of fg_burst
            norm_shifts: normalized shifts (-1 to 1)
    """

    def __init__(self,
                 samples_per_epoch=None,
                 bg_movement=True,
                 split='train',
                 shuffle=True,
                 num_fgs=None,
                 num_bgs=None,
                 border_crop=40):
        super().__init__()
        self.fg_dataset = MattingDataset(subset=split)
        self.num_fgs = num_fgs
        assert samples_per_epoch is None or samples_per_epoch <= len(
            self.fg_dataset)
        self.bg_dataset = ZurichRGBDataset(split=split,
                                           generate_burst=False,
                                           is_random_sample=shuffle,
                                           num_imgs=num_bgs)
        self.img2burst = GenerateBurst()

        self.fg_transform = PILToTensor()
        self.bg_transform = Compose([ToTensor(), RandomHorizontalFlip()])

        self.border_crop = border_crop
        self.samples_per_epoch = samples_per_epoch
        self.bg_movement = bg_movement
        self.split = split

    def __getitem__(self, index):
        if isinstance(self.num_fgs, int):
            index = index % self.num_fgs + 3

        bg_srgb = self.bg_transform(self.bg_dataset[index])
        bg_srgb = F.interpolate(
            bg_srgb.unsqueeze(0),
            scale_factor=1.5,
            mode='bilinear',
            align_corners=True)[0]  # base zurich images are too small
        bg_shape = bg_srgb.shape[-2:]

        fg_data = self.fg_dataset[index]
        fg_srgb = self.fg_transform(fg_data['image']) / 255.
        alpha_still = self.fg_transform(fg_data['alpha']).float() / 255.
        ratios = bg_shape[0] / fg_srgb.shape[-2], bg_shape[1] / fg_srgb.shape[
            -1]
        scale_factor = max(ratios)
        fg_srgb = F.interpolate(fg_srgb.unsqueeze(0),
                                scale_factor=scale_factor,
                                mode='bilinear',
                                align_corners=True).squeeze(0)
        alpha_still = F.interpolate(alpha_still.unsqueeze(0),
                                    scale_factor=scale_factor,
                                    mode='bilinear',
                                    align_corners=True).squeeze(0)

        crop_foreground = CenterCrop(bg_shape)
        fg_srgb = crop_foreground(fg_srgb)
        alpha_still = crop_foreground(alpha_still)
        fg_burst, fg_lrgb, fg_metadata = self.img2burst(fg_srgb,
                                                        alpha=alpha_still)
        if self.bg_movement:
            bg_burst, bg_lrgb, bg_metadata = self.img2burst(bg_srgb)
        else:
            self.img2burst.movement = False
            bg_burst, bg_lrgb, bg_metadata = self.img2burst(bg_srgb)
            self.img2burst.movement = True

        cropped_img_shape = [x - self.border_crop * 2 for x in bg_shape]
        alpha_burst = alpha_still.repeat(len(fg_burst), 1, 1, 1)
        crop_to_size = CenterCrop(cropped_img_shape)
        if len(fg_burst) > 1:
            shape_1 = torch.as_tensor([x - 1 for x in bg_shape]).reshape(
                1, 1, 1, 2) / 2
            normalized_shifts = (fg_metadata['shifts'].permute(0, 2, 3, 1) -
                                 shape_1) / shape_1
            fg_metadata['norm_shifts'] = crop_to_size(
                normalized_shifts.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            alpha_burst = F.grid_sample(alpha_burst,
                                        normalized_shifts.clamp(-1, 1),
                                        align_corners=True)
        else:
            fg_metadata.pop('shifts')

        fg_srgb = crop_to_size(fg_srgb)
        bg_srgb = crop_to_size(bg_srgb)
        fg_lrgb = crop_to_size(fg_lrgb)
        bg_lrgb = crop_to_size(bg_lrgb)
        fg_burst = crop_to_size(fg_burst)
        bg_burst = crop_to_size(bg_burst)
        alpha_burst = crop_to_size(alpha_burst)
        alpha_still = alpha_burst[0]
        comp_burst = fg_burst * alpha_burst + bg_burst * (1 - alpha_burst)
        comp_srgb = fg_srgb * alpha_still + bg_srgb * (1 - alpha_still)
        comp_lrgb = fg_lrgb * alpha_still + bg_lrgb * (1 - alpha_still)

        datapoint = {'metadata': fg_metadata, 'fg_path': fg_data['path']}
        for k in ('fg_srgb', 'bg_srgb', 'fg_lrgb', 'bg_lrgb', 'comp_srgb',
                  'comp_lrgb', 'fg_burst', 'bg_burst', 'alpha_burst',
                  'comp_burst'):
            if k in locals():
                datapoint[k] = locals()[k]
        return datapoint

    def __len__(self):
        if self.num_fgs is not None and self.split == 'test':
            return self.num_fgs
        elif self.samples_per_epoch is not None:
            return self.samples_per_epoch
        else:
            return len(self.fg_dataset)


@gin.configurable
class SyntheticVideoMattingDataset(torch.utils.data.Dataset):
    """Produces a linear RGB burst from a real video/burst foreground
    and a still background RGB image"""

    def __init__(self, fg_source, fg_subset, split='train', border_crop=40):
        super().__init__()
        self.fg_dataset = VideoMattingDataset(split=fg_subset,
                                              source=fg_source)
        self.bg_dataset = ZurichRGBDataset(split=split,
                                           generate_burst=False,
                                           is_random_sample=True)
        self.img2burst = GenerateBurst()
        self.fg_transform = PILToTensor()
        self.bg_transform = Compose([ToTensor(), RandomHorizontalFlip()])
        self.border_crop = border_crop

    def __getitem__(self, index):
        fg_data = self.fg_dataset[index]
        if fg_data is None:
            return

        bg_srgb = self.bg_transform(self.bg_dataset[index])
        bg_srgb = F.interpolate(
            bg_srgb.unsqueeze(0), scale_factor=1.5,
            align_corners=True)[0]  # base zurich images are too small
        bg_shape = bg_srgb.shape[-2:]

        fg_srgb = torch.stack(
            [self.fg_transform(frame) / 255. for frame in fg_data['video']])
        alpha_burst = torch.stack([
            self.fg_transform(frame).float() / 255.
            for frame in fg_data['alpha']
        ])
        ratios = bg_shape[0] / fg_srgb.shape[-2], bg_shape[1] / fg_srgb.shape[
            -1]
        a, b = min(ratios), max(ratios)
        scale_factor = b  #np.random.random() * (b-a) + a
        fg_srgb = F.interpolate(fg_srgb,
                                scale_factor=scale_factor,
                                mode='bilinear',
                                align_corners=True)
        alpha_burst = F.interpolate(alpha_burst,
                                    scale_factor=scale_factor,
                                    mode='bilinear',
                                    align_corners=True)

        crop_foreground = CenterCrop(bg_shape)
        fg_srgb = crop_foreground(fg_srgb)
        alpha_burst = crop_foreground(alpha_burst)

        fg_burst = []
        for frame in fg_srgb:
            fg_burst.append(
                self.img2burst.rgb2rawsingle(image=frame,
                                             randomize_gain=False))
        bg_burst, bg_lrgb, bg_metadata = self.img2burst(bg_srgb)

        fg_burst = torch.stack(fg_burst)
        normalized_shifts = bg_metadata['shifts'] - torch.as_tensor(
            bg_shape).reshape(1, 2, 1, 1) / 2
        normalized_shifts /= torch.as_tensor(bg_shape).reshape(1, 2, 1, 1) / 2
        # make foreground move together with background

        fg_burst = F.grid_sample(fg_burst,
                                 normalized_shifts.permute(0, 2, 3, 1),
                                 align_corners=True)
        alpha_burst = F.grid_sample(alpha_burst,
                                    normalized_shifts.permute(0, 2, 3, 1),
                                    align_corners=True)

        cropped_img_shape = [x - self.border_crop * 2 for x in bg_shape]
        crop_to_size = CenterCrop(cropped_img_shape)
        fg_srgb = crop_to_size(fg_srgb)
        bg_srgb = crop_to_size(bg_srgb)
        bg_lrgb = crop_to_size(bg_lrgb)
        fg_burst = crop_to_size(fg_burst)
        bg_burst = crop_to_size(bg_burst)
        alpha_burst = crop_to_size(alpha_burst)
        comp_burst = fg_burst * alpha_burst + bg_burst * (1 - alpha_burst)

        datapoint = {
            "fg_srgb_video": fg_srgb,
            "bg_srgb": bg_srgb,
            "fg_burst": fg_burst,
            "bg_burst": bg_burst,
            "alpha_burst": alpha_burst,
            "comp_burst": comp_burst,  # model input
        }
        return datapoint