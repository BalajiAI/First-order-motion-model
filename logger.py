"""For logging loss values and generated images"""

import logging
import numpy as np


class Logger:
    def __init__(self, log_path, log_filename):
        self.logger = self.get_logger(log_path, log_filename)
        self.batch_losses = []
        self.loss_names = None
  
    def get_logger(self, log_path, log_filename):
        logger = logging.getLogger()
        logger.handlers.clear()

        logger.setLevel(logging.INFO)    
        # Logging to a file
        file_handler = logging.FileHandler(f"{log_path}/{log_filename}")
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

        return logger    
        
    def log_batch_loss(self, losses):
        if self.loss_names is None:
            self.loss_names = list(losses.keys())
        self.batch_losses.append(list(losses.values()))

    def log_epoch_loss(self, epoch):
        loss_mean = np.array(self.batch_losses).mean(axis=0)

        loss_string = "; ".join([f"{name}-{value:.3f}" for name, value in zip(self.loss_names, loss_mean)])
        loss_string = "Epoch:" + str(epoch) + ' losses: ' + loss_string

        self.logger.info(loss_string)
        self.batch_losses = []
                