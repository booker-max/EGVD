from abc import ABC
import torch
from einops import rearrange
import numpy as np
import itertools, sys, importlib
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler

###DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.nn as nn

import sys
sys.path.append("/code/EGVD/network/utils")
from pytorch_ssim import *
sys.path.append("/code/EGVD/network/utils")
import utils_rmfd
from tools import batch_PSNR, batch_SSIM

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max = opt.epochs - opt.warmup_epochs, eta_min = opt.lr_min)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_cosine)
    elif opt.lr_policy == "constant":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def inputs2img(input):
    
    input = input.permute(1,2,0)

    target_images = input.clamp(0,1)
    target_images = target_images.detach().cpu().numpy()
    target_images = (target_images * 255).round().astype("uint8")

    return target_images

### 首尾
def eventsimg(input):

    input = input.cpu().numpy()

    pos_events = np.maximum(input, 0)
    neg_events = np.minimum(input, 0)

    pos_idx = np.nonzero(pos_events)
    neg_idx= np.nonzero(neg_events)
    nozero_index = np.nonzero(input)

    img_white = np.full((pos_events.shape[0], pos_events.shape[1], 3), fill_value=255, dtype="uint8")

    img_white[nozero_index[0], nozero_index[1], :] = 0
    img_white[pos_idx[0], pos_idx[1], 0] = 255
    img_white[neg_idx[0], neg_idx[1], -1] = 255

    return img_white

class EAVD(ABC):

    def __init__(self, args, local_rank):

        self.args = args
        self.local_rank = local_rank

        sys.path.append("/code/EGVD/network")
        baseline = importlib.import_module(self.args.ablation_type)
        RMFD = getattr(baseline, "RMFD")

        model_opts = vars(args)
        self.eavd = RMFD(model_opts, 10).to(self.local_rank)

        self.train_model_names = ["eavd"]
        self.eval_model_names = ["eavd"]
        self.optimizer_names = ["optimizer"]

        self.load_ddp()
        self.optimizers = []
        self.optimizer = torch.optim.Adam(
                                        itertools.chain(self.eavd.parameters()), 
                                        lr=args.learning_rate, 
                                        betas=(0.9, 0.999),
                                        eps=1e-8
                                        )
        self.optimizers.append(self.optimizer)

        self.schedulers = [get_scheduler(optimizer, self.args) for optimizer in self.optimizers]
        
        self.criterion = SSIM()

    def load_ddp(self):

        for name in self.train_model_names:

            if isinstance(name, str):

                setattr(self, name, torch.nn.SyncBatchNorm.convert_sync_batchnorm(getattr(self,name)).to(self.local_rank))
                setattr(self, name, DDP(getattr(self, name), device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False))
    
    def set_input(self, data):
        
        ### train集seq长度必须为7或者1
        self.frame_i, self.frame_g, self.frame_e = [], [], []
        for t in range(len(data)):
            self.frame_i.append(data[t]["frame"].to(self.local_rank))
            self.frame_g.append(data[t]["gt"].to(self.local_rank))
            self.frame_e.append(data[t]["events"].to(self.local_rank))

    def set_input_test(self, data):
        
        frame_i = data["rainy"]
        frame_g = data["gt"]
        frame_e = data["rainy_events"]
        
        self.inputs = torch.cat((frame_i[0], frame_i[1], frame_i[2]), dim=1).to(self.local_rank)
        self.gt = frame_g[1].to(self.local_rank)
        self.input_events = torch.cat((frame_e[0], frame_e[1]), dim=1).to(self.local_rank)

    def forward(self):

        lstm_state = None

        for t in range(0, self.args.sequence_length-2):

            frame_i1 = self.frame_i[t]
            frame_i2 = self.frame_i[t+1]
            frame_i3 = self.frame_i[t+2]

            frame_g1 = self.frame_g[t]
            frame_g2 = self.frame_g[t+1]
            frame_g3 = self.frame_g[t+2]

            event_i1 = self.frame_e[t]
            event_i2 = self.frame_e[t+1]

            inputs = torch.cat((frame_i1, frame_i2, frame_i3), dim=1)
            input_events = torch.cat((event_i1, event_i2), dim=1)
            
            frame_haze3_s1, lstm_state= self.eavd(inputs, lstm_state, input_events)
            lstm_state = utils_rmfd.repackage_hidden(lstm_state)
            
            self.optimizer.zero_grad()
            self.loss_sum = sum([-self.criterion(frame_haze3_s1[j], frame_g2.detach()) for j in range(len(frame_haze3_s1))])
            self.loss_sum.backward()
            self.optimizer.step()

        out = frame_haze3_s1[0]
        
        ## to do, 此时out, 和 self.gt都是五维向量
        self.psnr = batch_PSNR(out, frame_g2, ycbcr=False)
        self.ssim = batch_SSIM(out, frame_g2, ycbcr=False)
        
        ## to do
        ## 可视化batch_idx = 0, time_idx = seq_len//2
        self.vis_out = inputs2img(out[0,:,:,:]) 
        self.vis_gt = inputs2img(frame_g2[0,:,:,:])
        self.vis_input = inputs2img(frame_i2[0,:,:,:])

        self.vis_event_left = eventsimg(event_i1[0,-1,:,:]) #左边的event取最后一份
        self.vis_event_right = eventsimg(event_i2[0,0,:,:]) #右边的event取第一份

    def forward_test(self, lstm_state):

        frame_haze3_s1, lstm_state= self.eavd(self.inputs, lstm_state, self.input_events)
        lstm_state = utils_rmfd.repackage_hidden(lstm_state)

        out = frame_haze3_s1[0]
        self.loss_sum = sum([-self.criterion(frame_haze3_s1[j], self.gt.detach()) for j in range(len(frame_haze3_s1))])

        self.psnr = batch_PSNR(out, self.gt, ycbcr=False)
        self.ssim = batch_SSIM(out, self.gt, ycbcr=False)
        
        ## to do
        ## 可视化batch_idx = 0, time_idx = seq_len//2
        self.vis_out = inputs2img(out[0,:,:,:]) 
        self.vis_gt = inputs2img(self.gt[0,:,:,:])
        self.vis_input = inputs2img(self.inputs[0,3:6,:,:])

        self.vis_event_left = eventsimg(self.input_events[0,9,:,:]) #左边的event取最后一份
        self.vis_event_right = eventsimg(self.input_events[0,11,:,:]) #右边的event取第一份

        return lstm_state

    def update_learning_rate(self):

        for scheduler in self.schedulers:
            if self.args.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

    def get_losses(self):

        sum_loss = self.loss_sum.item()
        psnr = self.psnr
        ssim = self.ssim

        loss_record = {"sum": sum_loss}
        metrics_record = {"psnr":psnr, "ssim": ssim}
        
        return loss_record, metrics_record
    
    def train(self):
        for name in self.train_model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()
    
    def eval(self):

        for name in self.eval_model_names:

            if isinstance(name, str):

                net = getattr(self, name)

                net.eval()






