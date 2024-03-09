from argparse import ArgumentParser

from torch.utils.data import DataLoader

from dataset import VoxCelebaDataset, get_transform
from train_class import FirstOrderMotionModel


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="train_videos", help="path to training data")
    parser.add_argument("--config_path", default="config.yaml", help="path to config file")
    parser.add_argument("--log_path", default='logs', help="path to log")
    parser.add_argument("--checkpoint", default=None, help="path to save the checkpoint")

    args = parser.parse_args()

    dataset = VoxCelebaDataset(data_path=args.data_path, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = FirstOrderMotionModel(config_path=args.config_path,
                                  checkpoint_path=args.checkpoint_path)
    
    print("Training...")
    model.train(dataloader)
    