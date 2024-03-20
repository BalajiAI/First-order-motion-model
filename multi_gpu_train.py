import os
from argparse import ArgumentParser

from dataset import VideoDataset, DatasetRepeater
from augmentation import get_transform
from train_class import FirstOrderMotionModel

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank: int, world_size: int):
   """
   Args:
      rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12377"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    dataset = VideoDataset(data_path=args.data_path, id_sampling=True, transform=get_transform("voxceleb"))
    dataset = DatasetRepeater(dataset, num_repeats=75)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=DistributedSampler(dataset),
                            num_workers=2, pin_memory=True)
    model = FirstOrderMotionModel(config_path=args.config_path, log_path=args.log_path,
                                  checkpoint_path=args.checkpoint_path, gpu_id=rank)
    model.train(dataloader)
    destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="./moving-gif/train", help="path to training data")
    parser.add_argument("--config_path", default="./configs/voxceleb.yaml", help="path to config file")
    parser.add_argument("--log_path", default='logs', help="path to log")
    parser.add_argument("--checkpoint_path", default=None, help="path to save the checkpoint")

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
    