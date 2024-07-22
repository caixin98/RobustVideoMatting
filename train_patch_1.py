

from train_patch import train
from torch import multiprocessing as mp
import torch

            
if __name__ == '__main__':
    # config_file = "configs/train_base.gin"
    config_file = "configs/train_base_human_rvmnet.gin"
    print(config_file)
    print("Start training")
    # config_file = 'configs/train_overfitting_burst_eval.gin'
    # config_file = "configs/train_overfitting_burst_eval_rvmnet.gin"
    # trainer(0, world_size)
    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        nprocs=world_size,
        args=(world_size, config_file),
        join=True)