import argparse
import torch
import random
import os
from torch import nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms.functional import center_crop
from tqdm import tqdm
from dataset.synthetic_burst import SyntheticBurstDataset, SyntheticBurstPatch
from dataset.burst_eval import BurstEvalDataset
import gin
from train_loss import matting_loss, matting_loss_v2
from prefetch_generator import BackgroundGenerator
from torch.utils.data.dataloader import default_collate
from utils.image.rgb2raw import process_linear_image_raw

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def worker_init_fn(worker_id):
    device_id = worker_id % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

def recursive_collate(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: recursive_collate([d[key] for d in batch if key in d]) for key in elem}
    else:
        return default_collate(batch)

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  
    return recursive_collate(batch)

@gin.configurable()
class Trainer:
    def __init__(self, **kwargs):
        self.parse_args()
        self.config_kwargs(kwargs)

    def __call__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()


    def config_kwargs(self, kwargs):
        for k, v in kwargs.items():
            # print(f'Configuring {k} to {v}')
            setattr(self.args, k, v)



    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.log('Initializing distributed')
        os.environ['MASTER_ADDR'] = self.args.distributed_addr
        os.environ['MASTER_PORT'] = self.args.distributed_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def init_datasets(self):
        self.log('Initializing matting datasets')
        print("self.args.dataset", self.args.matting_subset)
        if self.args.dataset == 'synthetic':
            self.train_dataset = SyntheticBurstDataset(size=self.args.size, matting_subset = self.args.matting_subset, bg_subset = self.args.bg_subset, split = "train")
            self.validate_dataset = SyntheticBurstDataset(size=self.args.size, matting_subset = self.args.matting_subset, bg_subset = self.args.bg_subset, split = "val")
        elif self.args.dataset == "synthetic_patch":
            self.train_dataset = SyntheticBurstPatch(patch_size=self.args.patch_size, size=self.args.size, matting_subset = self.args.matting_subset, bg_subset = self.args.bg_subset, split = "train")
            self.validate_dataset = SyntheticBurstPatch(patch_size=self.args.patch_size, size=self.args.size, matting_subset = self.args.matting_subset, bg_subset = self.args.bg_subset, split = "val")
        elif self.args.dataset == "burst_eval":
            self.train_dataset = BurstEvalDataset(root=self.args.train_data)
            self.validate_dataset = BurstEvalDataset(root=self.args.val_data)
        
        self.train_sampler = DistributedSampler(self.train_dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=self.train_sampler, num_workers=self.args.num_workers, worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)
        self.validate_loader = DataLoader(self.validate_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, worker_init_fn=worker_init_fn, collate_fn=custom_collate_fn)
    def init_model(self):
        self.log('Initializing model')
        if self.args.model == 'bipnet':
            from model.bipnet_4d import BIPNet
            self.model = BIPNet()
        elif self.args.model == 'rvmnet':
            from model.model import MattingNetwork
            self.model = MattingNetwork(pretrained_backbone = True)
        elif self.args.model == 'burstnet':
            from model.matting_model import MattingModel
            self.model = MattingModel(temperature=self.args.temperature)

        if self.args.checkpoint:
            self.log(f'Restoring from checkpoint: {self.args.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=f'cuda:{self.rank}')))
        self.model = self.model.to(self.rank)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.scaler = GradScaler()

    
    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            self.writer = SummaryWriter(self.args.log_dir)

    def train(self):

        if not self.args.disable_validation:
            self.epoch = self.args.epoch_start
            self.step = self.args.epoch_start * len(self.train_loader)
            self.validate(sanity_check=True)

        self.log('Starting training')
        self.step = 0
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch
            self.step = epoch * len(self.train_loader)
            
          
            self.log(f'Training epoch: {epoch}')
            for data in tqdm(self.train_loader, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                # Low resolution pass
                self.train_mat(data, tag='patch')
                if self.step % self.args.checkpoint_save_interval == 0:
                    self.save()
                    
                self.step += 1

            if not self.args.disable_validation:
                self.validate()

            torch.cuda.empty_cache()
    
    def train_mat(self, data, tag):
        comp_burst = data['comp_burst'].to(self.rank, non_blocking=True)
        fgrs_linear = data['fgrs_linear'].to(self.rank, non_blocking=True)
        bgrs_linear = data['bgrs_linear'].to(self.rank, non_blocking=True)
        alpha_burst = data['alpha_burst'].to(self.rank, non_blocking=True)
        trimap_burst = data['trimap_burst'].to(self.rank, non_blocking=True)
        if "unprocess_metadata" in data:
                unprocess_metadata = data['unprocess_metadata']
        # print("comp_burst.shape, fgrs_linear.shape, bgrs_linear.shape, alpha_burst.shape, trimap_burst.shape",
                # comp_burst.shape, fgrs_linear.shape, bgrs_linear.shape, alpha_burst.shape, trimap_burst.shape)
        with autocast(enabled=not self.args.disable_mixed_precision):
    
            pred_fgr, pred_bgr, pred_pha = self.model_ddp(comp_burst, trimap_burst)

            if self.args.predict_seq:
                true_alpha = alpha_burst
                true_fgr = fgrs_linear
                true_bgr = bgrs_linear
                loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_alpha)
            else:
                true_alpha = alpha_burst[:,0:1]
                true_fgr = fgrs_linear[:,0:1]
                pred_fgr = pred_fgr[:,0:1]
                pred_pha = pred_pha[:,0:1]
                pred_bgr = pred_bgr[:,0:1]
                true_bgr = bgrs_linear[:,0:1]
                bgr_mask = (1 - alpha_burst.min(dim=1, keepdim=True).values).gt(0)
                loss = matting_loss_v2(pred_fgr, pred_bgr, pred_pha, true_fgr, true_bgr,  true_alpha, bgr_mask)

        self.scaler.scale(loss['bgr_l1']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        if self.rank == 0 and self.step % self.args.log_train_loss_interval == 0:
            for loss_name, loss_value in loss.items():
                if loss_name == 'bgr_l1':
                    self.writer.add_scalar(f'train_{tag}_{loss_name}', loss_value, self.step)
            
        if self.rank == 0 and self.step % self.args.log_train_images_interval == 0:
            if "unprocess_metadata" in data:
                pred_bgr = pred_bgr * bgr_mask
                true_bgr = true_bgr * bgr_mask

                comp_burst_center_frame = comp_burst[:, 0:1]
                comp_center_frame_gamma = process_linear_image_raw(comp_burst_center_frame.flatten(0, 1), unprocess_metadata, apply_demosaic=False)
                self.writer.add_image('train_comp_burst_gamma', make_grid(comp_center_frame_gamma, nrow=comp_center_frame_gamma.size(0)), self.step)
                pred_bgr_gamma = process_linear_image_raw(pred_bgr.flatten(0, 1), unprocess_metadata, apply_demosaic=False)
                self.writer.add_image('train_pred_bgr_gamma', make_grid(pred_bgr_gamma, nrow=pred_bgr_gamma.size(0)), self.step)
                true_bgr_gamma = process_linear_image_raw(true_bgr.flatten(0, 1), unprocess_metadata, apply_demosaic=False)
                self.writer.add_image('train_true_bgr_gamma', make_grid(true_bgr_gamma, nrow=true_bgr_gamma.size(0)), self.step)

            else:
                self.writer.add_image('train_pred_bgr', make_grid(pred_bgr.flatten(0, 1), nrow=pred_bgr.size(1)), self.step)
                self.writer.add_image('train_true_bgr', make_grid(true_bgr.flatten(0, 1), nrow=true_bgr.size(1)), self.step)
            
            self.writer.add_image('train_comp_burst', make_grid(comp_burst.flatten(0, 1), nrow=comp_burst.size(1)), self.step)
            

    def validate(self, sanity_check=False):
        if self.rank == 0:
            self.log(f'Validating at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            validate_step = 0
            validate_num = len(self.validate_loader)
            log_val_images_interval = validate_num // 10
            # print("validate_num", validate_num)
            # print("log_val_images_interval", log_val_images_interval)
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    for data in tqdm(self.validate_loader, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                        comp_burst = data['comp_burst'].to(self.rank, non_blocking=True)
                        fgrs_linear = data['fgrs_linear'].to(self.rank, non_blocking=True)
                        bgrs_linear = data['bgrs_linear'].to(self.rank, non_blocking=True)
                        alpha_burst = data['alpha_burst'].to(self.rank, non_blocking=True)
                        trimap_burst = data['trimap_burst'].to(self.rank, non_blocking=True)
                        if "unprocess_metadata" in data:
                            unprocess_metadata = data['unprocess_metadata']

                        pred_fgr, pred_bgr, pred_pha = self.model_ddp(comp_burst, trimap_burst)
                        if self.args.predict_seq:
                            true_alpha = alpha_burst
                            true_fgr = fgrs_linear
                            true_bgr = bgrs_linear
                            loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_alpha)
                        else:
                            true_alpha = alpha_burst[:,0:1]
                            true_fgr = fgrs_linear[:,0:1]
                            pred_fgr = pred_fgr[:,0:1]
                            pred_pha = pred_pha[:,0:1]
                            pred_bgr = pred_bgr[:,0:1]
                            true_bgr = bgrs_linear[:,0:1]
                            bgr_mask = 1 - alpha_burst.min(dim=1, keepdim=True).values
                            # print("bgr_mask.shape", bgr_mask.shape)
                            loss = matting_loss_v2(pred_fgr, pred_bgr, pred_pha, true_fgr, true_bgr, true_alpha, bgr_mask)
                        # print("true_alpha.shape, true_fgr.shape", true_alpha.shape, true_fgr.shape)
                        batch_size = comp_burst.size(0)
                        total_loss +=  loss['bgr_l1'].item() * batch_size
                        total_count += batch_size
                       
                        
                        # visualize validation images
                        if validate_step % log_val_images_interval == 0:
                            self.writer.add_image('val_pred_pha', make_grid(pred_pha.flatten(0, 1), nrow=pred_pha.size(0)), self.step + validate_step)
                            self.writer.add_image('val_true_alpha', make_grid(true_alpha.flatten(0, 1), nrow=true_alpha.size(0)), self.step + validate_step)
                            
                            if "unprocess_metadata" in data:
                                pred_bgr = pred_bgr * bgr_mask
                                true_bgr = true_bgr * bgr_mask
                                comp_burst_center_frame = comp_burst[:, 0:1]
                                comp_center_frame_gamma = process_linear_image_raw(comp_burst_center_frame.flatten(0, 1), unprocess_metadata, apply_demosaic=False)
                                self.writer.add_image('val_comp_burst_gamma', make_grid(comp_center_frame_gamma, nrow=comp_center_frame_gamma.size(0)), self.step + validate_step)
                                pred_bgr_gamma = process_linear_image_raw(pred_bgr.flatten(0, 1), unprocess_metadata, apply_demosaic=False)
                                self.writer.add_image('val_pred_bgr_gamma', make_grid(pred_bgr_gamma, nrow=pred_bgr_gamma.size(0)), self.step + validate_step)
                                true_bgr_gamma = process_linear_image_raw(true_bgr.flatten(0, 1), unprocess_metadata, apply_demosaic=False)
                                self.writer.add_image('val_true_bgr_gamma', make_grid(true_bgr_gamma, nrow=true_bgr_gamma.size(0)), self.step + validate_step)

                            else:
                                self.writer.add_image('val_pred_bgr', make_grid(pred_bgr.flatten(0, 1), nrow=pred_bgr.size(1)), self.step + validate_step)
                                self.writer.add_image('val_true_bgr', make_grid(true_bgr.flatten(0, 1), nrow=true_bgr.size(1)), self.step + validate_step)
                            self.writer.add_image('val_comp_burst', make_grid(comp_burst.flatten(0, 1), nrow=comp_burst.size(1)), self.step + validate_step)
                            
                        validate_step += 1
                        # print("validate_step", validate_step)
                        if sanity_check:
                            break

            avg_loss = total_loss / total_count
            self.writer.add_scalar('val_pha_loss', avg_loss, self.step)
            self.log(f'Validation set average loss: {avg_loss}')
        dist.barrier()
        


    def save(self):
        if self.rank == 0:
            os.makedirs(self.args.checkpoint_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint_dir, f'epoch-{self.epoch}.pth'))
            self.log('Model saved')
        dist.barrier()

    def cleanup(self):
        dist.destroy_process_group()

    def log(self, msg):
        print(f'[GPU{self.rank}] {msg}')

def train(rank, world_size, config_file = 'configs/train_base.gin'):
    gin.parse_config_file(config_file)
    # gin.parse_config_file(config_file)
    trainer = Trainer()
    print(f"Starting training on rank {rank} of {world_size}")
    trainer(rank, world_size)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/train_base.gin')
    args = parser.parse_args()
    print(args.config)
    return args
            
if __name__ == '__main__':
    # config_file = "configs/train_base.gin"
    # config_file = "configs/train_base_human.gin"
    # config_file = 'configs/train_overfitting_burst_eval.gin'
    # config_file = "configs/train_overfitting_burst_eval_rvmnet.gin"
    # print(config_file)
    print("Start training")
    args = parse_args()
    import sys
    sys.argv = [sys.argv[0]]
    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        nprocs=world_size,
        args=(world_size, args.config),
        join=True)