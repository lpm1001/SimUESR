import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from torchvision.models import vgg as vgg
import numpy as np
from math import exp
from basicsr.models.losses.loss_util import weighted_loss


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


    
class L_exp(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6, loss_weight=1.0):
        super(L_exp, self).__init__()
        self.loss_weight = loss_weight
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x):
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return torch.mean(d)*self.loss_weight
    
class L_TV(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L_TV, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.loss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    
class L_colorcos(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L_colorcos, self).__init__()
        self.loss_weight = loss_weight
    
    def forward(self, true_reflect, pred_reflect):
        b, c, h, w = true_reflect.shape
        true_reflect_view = true_reflect.view(b, c, h * w).permute(0, 2, 1)
        pred_reflect_view = pred_reflect.view(b, c, h * w).permute(0, 2, 1) # 16 x (512x512) x 3
        true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
        pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
        cose_value = true_reflect_norm * pred_reflect_norm
        cose_value = torch.sum(cose_value, dim=-1) # 16 x (512x512)  # print(cose_value.min(), cose_value.max())
        color_loss = torch.mean(1 - cose_value)
        return self.loss_weight*color_loss


class MSSSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, pred, target,window_size=11, size_average=True, val_range=None, normalize=True):
        device = pred.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        levels = weights.size()[0]
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = ssim(pred, target, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
            #print("sim",sim)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(pred, (2, 2))
            img2 = F.avg_pool2d(target, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        if normalize:
            mssim = (mssim + 1) / 2
            mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights
        output = torch.prod(pow1[:-1] * pow2[-1]) #返回所有元素的乘积
        return output
    

    
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None, loss_weight=1.0):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.loss_weight = loss_weight
   
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
   
    def forward(self, pred, target):
        (_, channel, _, _) = pred.size()
   
        if channel == self.channel and self.window.dtype == pred.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(pred.device).type(pred.dtype)
            self.window = window
            self.channel = channel
  
        return self.loss_weight*ssim(pred, target, window=window, window_size=self.window_size, size_average=self.size_average)

def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # import ipdb
    # ipdb.set_trace()
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    # import ipdb
    # ipdb.set_trace()
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel) # 高斯滤波 求均值
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel) # 求均值

    mu1_sq = mu1.pow(2) # 平方
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq # var(x) = Var(X)=E[X^2]-E[X]^2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2 # 协方差

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret
##############################################################################################################################