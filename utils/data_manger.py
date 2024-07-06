import json
import os
import glob

def remove_last_100(img_paths, mask_paths):
    return img_paths[:-100], mask_paths[:-100]
def take_last_100(img_paths, mask_paths):   
    return img_paths[-100:], mask_paths[-100:]

class DatasetManager:
    def __init__(self, data_key = None, config_path='dataset/datasets_config.json'):
        self.config_path = config_path
        self.dataset_configs = self.load_config()
        self.full_config = self.load_config()  # Keep a copy of the full config
        # pls refer the datasets_config.json to see the structure of the config file, and the data_key is like
        # "fg_phas", "bg_imgs", "video_matting"
        self.dataset_configs = self.full_config[data_key] if data_key is not None else self.full_config

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return json.load(file)

    def save_config(self):
        with open(self.config_path, 'w') as file:
            json.dump(self.full_config, file, indent=4)

    def add_dataset(self, dataset_dict):
        self.dataset_configs.update(dataset_dict)
        self.save_config()
        
    def remove_dataset(self, key):
        if key in self.dataset_configs:
            del self.dataset_configs[key]
            self.save_config()

    def list_datasets(self):
        return list(self.dataset_configs.keys())
    
    def str_to_function(self,function_str):
    # Use eval() or exec() to convert the string to a function object
    # ...but be careful, this can be dangerous if the string is from an untrusted source
        return eval(function_str)
    
    #give the key of the dataset and get the image paths
    def get_paths(self, key):
        raise NotImplementedError

class MattingDatasetManager(DatasetManager):
    
    def __init__(self, data_key = "fg_phas", config_path='dataset/datasets_config.json'):
        super().__init__(data_key, config_path)
        
    def get_paths(self, key):
        if key not in self.dataset_configs:
            raise ValueError(f"Dataset {key} not found.")
        
        config = self.dataset_configs[key]
        data_dir = os.path.join(self.full_config["data_path"], config['path'])
        img_pattern = config['img_pattern']
        mask_pattern = config['mask_pattern']
        preposecssing = config['preprocessing']
        img_paths = sorted(glob.glob(os.path.join(data_dir, img_pattern)))
        alpha_paths = sorted(glob.glob(os.path.join(data_dir, mask_pattern)))
        if preposecssing is not None:
            preposecssing_func = self.str_to_function(preposecssing)
            img_paths, alpha_paths = preposecssing_func(img_paths, alpha_paths)

        assert len(img_paths) > 0 and os.path.exists(img_paths[0]), f'No images found in {os.path.join(data_dir, img_pattern)}'
        assert len(alpha_paths) > 0 and os.path.exists(alpha_paths[0]), 'No masks found'
        assert len(img_paths) == len(alpha_paths), 'Number of images and masks do not match'
        return img_paths, alpha_paths

class VideoMattingManager(DatasetManager):
        
        def __init__(self, data_key = None, config_path='dataset/datasets_config.json'):
            super().__init__(data_key, config_path)
        # To be implemented
        def get_paths(self, key):
         
            raise NotImplementedError


class BackgroundDatasetManager(DatasetManager):
    
    def __init__(self, data_key = "bg_imgs", config_path='dataset/datasets_config.json'):
        super().__init__(data_key, config_path)

    def get_paths(self, key):
        if key not in self.dataset_configs:
            raise ValueError(f"Dataset {key} not found.")
        
        config = self.dataset_configs[key]
        data_dir = os.path.join(self.full_config["data_path"], config['path'])
        img_pattern = config['img_pattern']
        preposecssing = config['preprocessing']
        img_paths = sorted(glob.glob(os.path.join(data_dir, img_pattern)))
                           
        if preposecssing is not None:
            preposecssing_func = self.str_to_function(preposecssing)
            img_paths = preposecssing_func(img_paths)

        assert len(img_paths) > 0 and os.path.exists(img_paths[0]), f'No images found in {os.path.join(data_dir, img_pattern)}'
        return img_paths


if __name__ == '__main__':
    manager = MattingDatasetManager()
    print(manager.list_datasets())
    img_paths, mask_paths = manager.get_paths('high_res')
    print(img_paths, mask_paths)