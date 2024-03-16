import os
import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from models.keypoint_detector import KPDetector
from models.generator import OcclusionAwareGenerator
from models.discriminator import MultiScaleDiscriminator
from models.model import GeneratorModel, DiscriminatorModel

from logger import Logger

from torch.nn.parallel import DistributedDataParallel as DDP


class FirstOrderMotionModel:
    def __init__(self, config_path, log_path="logs", checkpoint_path=None, gpu_id=None) -> None:
        self.config = self._read_config(config_path)

        self.gpu_id = gpu_id

        self.initialize_models(self.config['model_params'])
        self.initialize_optimizers(self.config['train_params'])
        
        if checkpoint_path is not None:
            self.start_epoch = self.load_checkpoint(checkpoint_path)
        else:
            self.start_epoch = 0

        self.initialize_lr_schedulers(self.config['train_params'], self.start_epoch)

        if not os.path.exists(log_path):
            os.makedirs(log_path+'/checkpoints')
            os.makedirs(log_path+'/visualizations')       

        self.logger = Logger(log_path, "log.txt")
        self.log_path = log_path

    def _read_config(self, config_path):
        f = open(config_path, "r")
        config = yaml.safe_load(f)
        return config        

    def initialize_models(self, model_params):
        self.kp_detector = KPDetector(**model_params['kp_detector_params'],
                                      **model_params['common_params'])
        self.kp_detector = nn.SyncBatchNorm.convert_sync_batchnorm(self.kp_detector)
        self.kp_detector = DDP(self.kp_detector, device_ids=[self.gpu_id])
        self.kp_detector = torch.compile(self.kp_detector)
        
        self.generator = OcclusionAwareGenerator(**model_params['generator_params'],
                                                 **model_params['common_params'])
        self.generator = nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
        self.generator = DDP(self.generator, device_ids=[self.gpu_id])
        self.generator = torch.compile(self.kp_detector)        
        
        self.discriminator = MultiScaleDiscriminator(**model_params['discriminator_params'],
                                                     **model_params['common_params']).to(self.device)
        self.discriminator =  DDP(self.discriminator, device_ids=[self.gpu_id])
        self.discriminator = torch.compile(self.discriminator)

        self.generator_model = GeneratorModel(self.kp_detector, self.generator,
                                              self.discriminator, self.config['train_params'], self.gpu_id)
        
        self.discriminator_model = DiscriminatorModel(self.kp_detector, self.generator,
                                                      self.discriminator, self.config['train_params'], self.gpu_id)
        
    def initialize_optimizers(self, train_params):
        self.opt_kp_detector = torch.optim.Adam(self.kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
        self.opt_generator = torch.optim.Adam(self.generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))

    def initialize_lr_schedulers(self, train_params, start_epoch=0):
        self.scheduler_kp_detector = MultiStepLR(self.opt_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                                 last_epoch= -1 + start_epoch * (train_params['lr_kp_detector'] != 0))
        self.scheduler_generator = MultiStepLR(self.opt_generator, train_params['epoch_milestones'], gamma=0.1,
                                               last_epoch= start_epoch - 1)
        self.scheduler_discriminator = MultiStepLR(self.opt_generator, train_params['epoch_milestones'], gamma=0.1,
                                                  last_epoch= start_epoch - 1)

    def save_checkpoint(self, current_epoch):
        cpk = {'kp_detector': self.kp_detector.module.state_dict(),
               'generator': self.generator.module.state_dict(),
               'discriminator': self.discriminator.module.state_dict(),
               'opt_kp_detector': self.opt_kp_detector.module.state_dict(),
               'opt_generator': self.opt_generator.module.state_dict(),
               'opt_discriminator': self.opt_discriminator.module.state_dict(),
               'epoch': current_epoch}
        cpk_path = f"{self.log_path}/checkpoints/{current_epoch}-checkpoint.pth"
        torch.save(cpk, cpk_path)          

    def load_checkpoint(self, checkpoint_path):
        cpk = torch.load(checkpoint_path, map_location=self.device)
        self.kp_detector.load_state_dict(cpk['kp_detector'])
        self.generator.load_state_dict(cpk['generator'])
        self.discriminator.load_state_dict(cpk['discriminator'])
        self.opt_kp_detector.load_state_dict(cpk['opt_kp_detector'])
        self.opt_generator.load_state_dict(cpk['opt_generator'])
        self.opt_discriminator.load_state_dict(cpk['opt_discriminator'])

        return cpk['epoch']

    def optimize(self, x):
        losses_generator, generated = self.generator_model(x)

        loss_values = [val.mean() for val in losses_generator.values()]
        loss = sum(loss_values)
        
        #update kp_detector & generator
        self.opt_kp_detector.zero_grad()
        self.opt_generator.zero_grad()
        loss.backward()
        self.opt_kp_detector.step()
        self.opt_generator.step()

        if self.config['train_params']['loss_weights']['generator_gan'] != 0:

            losses_discriminator = self.discriminator_model(x, generated)
            loss_values = [val.mean() for val in losses_discriminator.values()]
            loss = sum(loss_values)
            
            #update discriminator
            self.opt_discriminator.zero_grad()
            loss.backward()
            self.opt_discriminator.step()

        else:
            losses_discriminator = {}

        losses_generator.update(losses_discriminator)
        losses = {key: value.mean().detach().data.cpu().item() for key, value in losses_generator.items()}
        
        return losses, generated

    def train(self, dataloader):
        for epoch in range(self.start_epoch, self.config['train_params']['num_epochs']):
            dataloader.sampler.set_epoch(epoch)
            for x in dataloader:
                x = {'source': x['source'].to(self.device),
                     'driving': x['driving'].to(self.device)}
                losses, output = self.optimize(x)
                self.logger.log_batch_loss(losses)
            
            #update lr
            self.scheduler_kp_detector.step()
            self.scheduler_generator.step()
            self.scheduler_discriminator.step()

            self.logger.log_epoch_loss(epoch)
     
            if ((epoch + 1) % self.config['train_params']['checkpoint_freq'] == 0) and (self.gpu_id == 0):
                self.save_checkpoint(epoch)

            if (epoch + 1) % self.config['train_params']['visualization_freq'] == 0 and (self.gpu_id == 0):
                self.logger.log_vis_images(x['source'], x['driving'], output, epoch)
