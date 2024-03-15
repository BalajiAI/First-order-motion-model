import os
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image, ImageSequence

from augmentation import get_transform


def read_video(path:str):
    """
    Reads the video from the given path.
    The path could contain any of the following:
        - .mp4 file or any video file format
        - folder of images
        - .jpg or any image format which contains concatenated frames
    """

    if path.endswith('.gif'):       
        image = Image.open(path)
        frames = []
        for frame in ImageSequence.Iterator(image):
            frame = frame.convert('RGB')  
            frames.append(np.asarray(frame))
        video_arr =  np.stack(frames, axis=0)

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

        video_arr = torch.tensor(video_arr) / 255
        video_arr = video_arr.permute(0, 3, 1, 2)

        if self.transform is not None:
            video_arr = self.transform(video_arr)

        source, driving = video_arr[0], video_arr[1]

        output = {"source": source,
                  "driving": driving}
        
        return output
