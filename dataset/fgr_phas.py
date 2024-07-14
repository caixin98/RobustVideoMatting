# obtain the original foreground and alpha images from the collected datasets
import glob
import os
import random
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset
from PIL import Image
import sys
# sys.path.append('..')
from utils.data_manger import MattingDatasetManager # type: ignore



subset_type = Tuple[Tuple[Union[int, str]], Union[int, str]]

class ForegroundAlphaDataset(Dataset):
    subset_mapping = {
        0: 'high_res',
        1: 'aim_train',
        2: 'aim_test',
        3: "distinctions_train",
        4: "distinctions_test",
        5: "distinctions_sample_train",
    }
    data_manager = MattingDatasetManager()
    def __init__(self, train_test_subset: subset_type, split: str = "train", samples_per_epoch: int = None):
        super().__init__()
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.setup_paths(train_test_subset)

    def setup_paths(self, train_test_subset: subset_type):
        self.img_paths = []
        self.alpha_paths = []
        if self.split == 'train':
            for ix in train_test_subset[0]:
                img_paths, alpha_paths = self.get_paths(ix)
                self.img_paths += img_paths
                self.alpha_paths += alpha_paths
        elif self.split in ('val', 'test'):
            self.img_paths, self.alpha_paths = self.get_paths(train_test_subset[1])
        else:
            raise ValueError(f"Invalid split {self.split}. Expected 'train', 'val', or 'test'.")

    def get_paths(self, subset: Union[int, str]) -> Tuple[List[str], List[str]]:
        if isinstance(subset, str):
            dataset_name = subset
        else:
            dataset_name = self.subset_mapping[subset]
        img_paths, alpha_paths = self.data_manager.get_paths(dataset_name)

        if not img_paths or not os.path.exists(img_paths[0]):
            raise FileNotFoundError(f'No images found in {dataset_name}')
        if not alpha_paths or not os.path.exists(alpha_paths[0]):
            raise FileNotFoundError('No masks found')

        return img_paths, alpha_paths

    
    def __getitem__(self, index: int) -> Dict[str, Image.Image]:
        path = self.img_paths[index]
        image = Image.open(path).convert('RGB')
        alpha = Image.open(self.alpha_paths[index]).convert('L')
        
        return {'image': image, 'alpha': alpha, 'path': path}

    def __len__(self) -> int:
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.img_paths)
    

    