from argparse import ArgumentParser

from torch.utils.data import DataLoader

from dataset import VideoDataset, DatasetRepeater
from augmentation import get_transform
from train_class import FirstOrderMotionModel


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="./moving-gif/train", help="path to training data")
    parser.add_argument("--config_path", default="./configs/mgif.yaml", help="path to config file")
    parser.add_argument("--log_path", default='logs', help="path to log")
    parser.add_argument("--checkpoint_path", default=None, help="path to save the checkpoint")

    args = parser.parse_args()

    dataset = VideoDataset(data_path=args.data_path, id_sampling=False, transform=get_transform("mgif"))
    dataset = DatasetRepeater(dataset, num_repeats=25)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    model = FirstOrderMotionModel(config_path=args.config_path,
                                  log_path=args.log_path,
                                  checkpoint_path=args.checkpoint_path)
    
    print("Training...")
    model.train(dataloader)
    