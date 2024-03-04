"""Testing code for all the models"""

import torch

from hourglass import Hourglass
from keypoint_detector import KPDetector
from dense_motion_net import DenseMotionNetwork
from generator import OcclusionAwareGenerator
from discriminator import MultiScaleDiscriminator


if __name__ == "__main__":
   
   #Test Hourglass
   sample_input = torch.randn((1, 3, 256, 256))
   hourglass = Hourglass(block_expansion=32, in_features=3, num_blocks=3, max_features=256)
   print(hourglass(sample_input).shape)

   #Test KPDetector
   kp_detector = KPDetector(block_expansion=32, num_kp=10, num_channels=3, max_features=256,
                 num_blocks=5, temperature=0.1, estimate_jacobian=True, scale_factor=0.25,
                 single_jacobian_map=False, pad=0)
   out_kp = kp_detector(sample_input)
   print(out_kp['value'].shape)
   print(out_kp['jacobian'].shape)

   #Test DenseMotionNetwork
   dense_motion_net = DenseMotionNetwork(block_expansion=64, num_blocks=5, max_features=1024,
                                         num_kp=10, num_channels=3, estimate_occlusion_map=True,
                                         scale_factor=0.25)
   source_image = torch.randn((1, 3, 256, 256))
   kp_driving = out_kp
   kp_source = out_kp
   out = dense_motion_net(source_image, kp_driving, kp_source)
   print(out['occlusion_map'].shape)

   #Test OcclusionAwareGenerator
   dense_motion_params = {'block_expansion': 64, 'max_features': 1024,
                          'num_blocks': 5, 'scale_factor': 0.25}
   generator = OcclusionAwareGenerator(num_channels=3, num_kp=10, block_expansion=64,
                                       max_features=512, num_down_blocks=2, num_bottleneck_blocks=6,
                                       estimate_occlusion_map=True, dense_motion_params=dense_motion_params, estimate_jacobian=True)
   out_gen = generator(source_image, kp_driving, kp_source)
   print(out_gen['prediction'].shape)
   
   #Test MultiScaleDiscriminator
   discriminator = MultiScaleDiscriminator(scales=[1], block_expansion=32, max_features=512,
                                           num_blocks=4, sn=True)
   #Discriminator takes image_pyramid output as an input

   
