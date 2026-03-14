import pyiqa

from basicsr.utils import img2tensor, tensor2img
from torch.utils import data as data
import glob
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys

class AdaptiveReward(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda")
        self.iqa_psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        self.iqa_maniqa = pyiqa.create_metric('maniqa', device=device)
        self.iqa_musiq = pyiqa.create_metric('musiq', device=device)

    def normalize_tensor(self, tensor):

        min_val = tensor.min()
        max_val = tensor.max()
        
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def forward(self, x, y, fedilty_ratio, detail_ratio):
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        b,gs,c,h,w=x.shape
        reward = torch.zeros([b,gs])
        for i in range(b):
            fedilty_i = fedilty_ratio[i]
            detail_i = detail_ratio[i]
            x_i = x[i]
            y_i = y[i] 
            psnr_result = self.normalize_tensor(self.iqa_psnr(x_i, y_i))
            musiq_result = self.normalize_tensor(self.iqa_musiq(x_i).squeeze(1))
            maniqa_result = self.normalize_tensor(self.iqa_maniqa(x_i).squeeze(1))

            reward_i = fedilty_i*psnr_result + detail_i*0.5*(maniqa_result+musiq_result)
            combined_mean = torch.mean(reward_i)
            combined_std = reward_i.std(unbiased=True) 
            reward_i = (reward_i - combined_mean) / (combined_std+1e-8)
            reward[i] = reward_i        
        
        return reward.detach()

