# evaluation dataset for burst image matting
# load the pre-generated burst data and evaluate the performance of the model
from PIL import Image
import os 
from torch.utils.data import Dataset
from torchvision import transforms
import torch
class BurstEvalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.data = self.load_data()
    def load_data(self):
        data = []
        for i in os.listdir(self.root):
            data_dir = os.path.join(self.root, i)
            pha, fgr, bgr, com, tri = [], [], [], [], []
            len_burst = len(os.listdir(os.path.join(data_dir, 'pha')))
            for j in range(len_burst):
                pha.append(os.path.join(data_dir, 'pha', f'{j:04}.png'))
                fgr.append(os.path.join(data_dir, 'fgr', f'{j:04}.png'))
                bgr.append(os.path.join(data_dir, 'bgr', f'{j:04}.png'))
                com.append(os.path.join(data_dir, 'com', f'{j:04}.png'))
                tri.append(os.path.join(data_dir, 'tri', f'{j:04}.png'))

            data.append({'alpha_burst': pha, 'fgrs_linear': fgr, 'bgrs_linear': bgr, 'comp_burst': com, 'trimap_burst': tri})
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        for key in sample:
            sample[key] = [Image.open(img) for img in sample[key]]
            if key == 'alpha_burst':
                sample[key] = [img.convert('L') for img in sample[key]]
            sample[key] = torch.stack([self.transform(img) for img in sample[key]])
        return sample