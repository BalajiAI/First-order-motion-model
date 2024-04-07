import os
import yaml
from argparse import ArgumentParser

import numpy as np
from scipy.spatial import ConvexHull
import cv2
import torch
import torchvision
from torchvision.transforms import v2

from dataset import read_video
from models.keypoint_detector import KPDetector
from models.generator import OcclusionAwareGenerator


def load_checkpoint(config_path, checkpoint_path, device):

    f = open(config_path, "r")
    config = yaml.safe_load(f)
    model_params = config['model_params']  

    #initialize models
    kp_detector = KPDetector(**model_params['kp_detector_params'],
                            **model_params['common_params']).to(device)
    
    generator = OcclusionAwareGenerator(**model_params['generator_params'],
                                        **model_params['common_params']).to(device)
    
    #update model's state_dict
    ckp = torch.load(checkpoint_path, map_location=device)

    kp_detector.load_state_dict(ckp['kp_detector'])
    generator.load_state_dict(ckp['generator'])

    kp_detector.eval()
    generator.eval()

    return kp_detector, generator


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].detach().cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].detach().cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, device='cpu'):
    with torch.no_grad():
        predictions = []
        #source = torch.tensor(source_image, dtype=torch.float32) 
        #source = source / 255
        #source = source.permute(2, 0, 1)
        source = source.unsqueeze(0)
        source = source.to(device)

        #driving = torch.tensor(driving_video, dtype=torch.float32) 
        #driving = driving / 255
        #driving = driving.permute(0, 3, 1, 2)
        #driving = driving.to(device)

 
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[0].unsqueeze(0))

        for frame_idx in range(driving.shape[0]):
            driving_frame = driving[frame_idx]
            driving_frame = transforms(driving_frame)
            driving_frame = driving.unsqueeze(0)
            driving_frame = driving_frame.to(device)

            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            prediction = out['prediction'][0].permute(1, 2, 0)
            prediction = prediction.detach().cpu().numpy()
            prediction = prediction * 255
            prediction = prediction.astype(np.uint8)

            predictions.append(prediction)

        predictions = np.stack(predictions, axis=0)
    
    return predictions


def save_video(source, driving, prediction):

    source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)

    im_h, im_w = (256, 256)
    border_width = 1
    border_color = (0, 0, 0)    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 10.0, (im_w * 3, im_h))  

    # Write each frame to the output video
    for i in range(driving.shape[0]):
        driving_frame = cv2.cvtColor(driving[i], cv2.COLOR_RGB2BGR)
        prediction_frame = cv2.cvtColor(prediction[i], cv2.COLOR_RGB2BGR)        
        combined_video = np.concatenate((source, driving_frame, prediction_frame), axis=1)
        combined_video[:, im_w - border_width:im_w + border_width, :] = border_color
        combined_video[:, 2 * im_w - border_width:2 * im_w + border_width, :] = border_color
        out.write(combined_video)

    out.release()
    print("video file has saved.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_path", default="./mgif_test/00065.gif", help="path to source video")
    parser.add_argument("--driving_path", default="./mgif_test/00017.gif", help="path to driving video")    
    parser.add_argument("--config_path", default="./configs/mgif.yaml", help="path to config file")
    parser.add_argument("--checkpoint_path", default=None, help="path to save the checkpoint")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    kp_detector, generator = load_checkpoint(args.config_path, args.checkpoint_path, device)
    source = read_video(args.source_path)[0]
    driving = read_video(args.driving_path)

    transforms = v2.Compose([v2.ToDtype(torch.uint8, scale=True),                       
                        v2.Resize(256, antialias=True),
                        v2.ToDtype(torch.float32, scale=True),])
    source = transforms(source)
    
    predictions = make_animation(source, driving, generator, kp_detector)

    save_video(source, driving, predictions)
    
