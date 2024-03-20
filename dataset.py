import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image, ImageSequence
import torchvision

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

    elif path.endswith('.mp4'):
        video_arr = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        video_arr = video_arr.numpy()

    return video_arr


class VideoDataset(Dataset):
    def __init__(self, data_path:str, id_sampling=False, transform=None):
        super().__init__()
        self.data_path = data_path
        self.id_sampling = id_sampling
        self.transform = transform
        self.videos = os.listdir(data_path)
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx:int):
        if self.id_sampling:
            id_name = self.videos[idx]
            id_videos = glob.glob(f"{self.data_path}/{id_name}/**/*.mp4")
            video_path = np.random.choice(id_videos)
            video_arr = read_video(video_path)
        else:
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


class DatasetRepeater(Dataset):
    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
    