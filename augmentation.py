import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


class RandomTimeFlip(v2.Transform):
    def __init__(self, p:float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            x = torch.stack((x[1], x[0]))       
        return x


def get_transform(dataset_name):
    if dataset_name == "mgif":
        transforms = v2.Compose([
                                v2.ToDtype(torch.uint8, scale=True),
                                RandomTimeFlip(0.5),
                                v2.RandomHorizontalFlip(0.5),
                                v2.RandomCrop(256),
                                v2.ColorJitter(hue=0.5),
                                v2.ToDtype(torch.float32, scale=True),])
                                #v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    elif dataset_name == "voxceleb":
        transforms = v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.uint8, scale=True),
                        v2.Resize([256, 256], antialias=True),
                        v2.RandomHorizontalFlip(0.5),
                        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        v2.ToDtype(torch.float32, scale=True),])
                        #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    return transforms
