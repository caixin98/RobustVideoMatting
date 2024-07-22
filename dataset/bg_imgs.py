# obtain the background images from the collected datasets

import glob
import os
import random
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset
from PIL import Image
import sys
# sys.path.append('..')
from utils.data_manger import BackgroundDatasetManager # type: ignore
import json, h5py



subset_type = Tuple[Tuple[Union[int, str]], Union[int, str]]

class BackgroundImageDataset(Dataset):
    subset_mapping = {
        0: 'bg_20k_train',
        1: 'bg_20k_test',
        2: 'bg_20k_sample_train',
        3: 'bg_20k_sample_test',
    }
    def __init__(self, train_test_subset: subset_type, split: str = "train", samples_per_epoch: int = None):
        super().__init__()
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.data_manager = BackgroundDatasetManager()
        self.setup_paths(train_test_subset)

    def setup_paths(self, train_test_subset: subset_type):
        self.img_paths = []
        self.alpha_paths = []
        if self.split == 'train':
            for ix in train_test_subset[0]:
                img_paths = self.get_paths(ix)
                self.img_paths += img_paths
            
        elif self.split in ('val', 'test'):
            self.img_paths = self.get_paths(train_test_subset[1])
        else:
            raise ValueError(f"Invalid split {self.split}. Expected 'train', 'val', or 'test'.")

    def get_paths(self, subset: Union[int, str]) -> Tuple[List[str], List[str]]:
        if isinstance(subset, str):
            dataset_name = subset
        else:
            dataset_name = self.subset_mapping[subset]
        img_paths = self.data_manager.get_paths(dataset_name)

        if not img_paths or not os.path.exists(img_paths[0]):
            raise FileNotFoundError(f'No images found in {dataset_name}')
 
        return img_paths
    
    def __getitem__(self, index: int) -> Dict[str, Image.Image]:
        path = self.img_paths[index]
        image = Image.open(path).convert('RGB')
        return {'image': image, 'is_raw': False}

    def __len__(self) -> int:
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.img_paths)

# datasets/Zurich-RAW-to-DSLR-Dataset-canon-{split}.hdf5
class ZurichRGBDataset(Dataset):
    def __init__(self, config_path='dataset/datasets_config.json', split: str = "train", samples_per_epoch: int = None):
        super().__init__()
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.config_path = config_path
        self.full_config = self.load_config()
        self.config = self.full_config['zurich_raw']
        self.img_paths = os.path.join(self.full_config['data_path'], self.config[split])
        

        # with h5py.File(self.img_paths, 'r') as f:
            # self.images = f['image']
        self.images = h5py.File(self.img_paths, 'r')['image']

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return json.load(file)

    def __getitem__(self, index: int) -> Dict[str, Image.Image]:
        image = Image.fromarray(self.images[index])
        return {'image': image, 'is_raw': False}
    
    def __len__(self) -> int:
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.img_paths)
        

