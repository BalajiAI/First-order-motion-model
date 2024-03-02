"""Testing code for all the models"""

import torch

from hourglass import Encoder
from hourglass import Decoder
from hourglass import Hourglass

if __name__ == "__main__":
   
   #Test Hourglass
   sample_input = torch.randn((1, 3, 256, 256))
   encoder = Encoder(block_expansion=32, in_features=3, num_blocks=3, max_features=256)
   decoder = Decoder(block_expansion=32, in_features=3, num_blocks=3, max_features=256)
   hourglass = Hourglass(block_expansion=32, in_features=3, num_blocks=3, max_features=256)
   print(hourglass(sample_input).shape)
