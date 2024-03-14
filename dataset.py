import os
import numpy as np
import torch
from torch.utils.data import Dataset

from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread

import torchvision
from torchvision.transforms import v2


def get_transform():
    transforms = v2.Compose([
                             v2.ToImage(),
                             v2.ToDtype(torch.uint8, scale=True),
                             v2.Resize([256, 256], antialias=True),
                             v2.RandomHorizontalFlip(0.5),
                             v2.ColorJitter(brightness=0.1, saturation=0.1, contrast=0.1, hue=0.1),
                             v2.ToDtype(torch.float32, scale=True),])
                             #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms


def read_video(path:str, frame_shape=(256, 256, 3)):
    """
    Reads the video from the given path.
    The path could contain any of the following:
        - .mp4 file or any video file format
        - folder of images
        - .jpg or any image format which contains concatenated frames
    """

    if path.endswith('.mp4') or path.endswith('.gif'):        
        video = np.array(mimread(path))

        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        #video_arr = img_as_float32(video)
        video_arr = video        

    return video_arr


class VideoDataset(Dataset):
    def __init__(self, data_path:str, transform=None):
        super().__init__()
        self.data_path = data_path
        self.videos = os.listdir(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx:int):
        video_name = self.videos[idx]
        video_arr = read_video(f"{self.data_path}/{video_name}")

        num_frames = video_arr.shape[0]
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
        video_arr = video_arr[frame_idx]

        source, driving = video_arr[0], video_arr[1]

        if self.transform is not None:
            source = self.transform(source)
            driving = self.transform(driving)

        output = {"source": source,
                  "driving": driving}
        
        return output
