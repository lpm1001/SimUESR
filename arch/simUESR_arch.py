# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stx
import math
from ipdb import set_trace as st
##########################################################################
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x
    


    
class TGMMDRM(nn.Module):
    def __init__(self, n_feat, n_MDRM, height, width, chan_factor, bias=False, groups=1):
        super(TGMMDRM, self).__init__()
        self.n_feat = n_feat
        modules_body = []
        for i in range(2*n_MDRM):
            if (i%2) == 0:
                modules_body.append(TGM(n_feat))
            else:
                modules_body.append(MDRM(n_feat, height, width, chan_factor, bias, groups))
        self.body = nn.Sequential(*modules_body)
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, x):
        res = self.body(x)
        res = self.conv(res[:,:self.n_feat,:,:])
        res += x[:,:self.n_feat,:,:]
        return torch.cat((res, x[:,self.n_feat:,:,:]), dim=1)


##########################################################################

class SimUESR(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        n_MDRM=2,
        height=3,
        width=2,
        scale=1,
        bias=False,
        task= None
    ):
        super(SimUESR, self).__init__()

        kernel_size=3
        self.n_feat = n_feat
        self.task = task
        
        self.scale = scale
        if scale == 2:
            inp_channels = inp_channels * 4
        elif scale == 1:
            inp_channels = inp_channels * 16

        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)
        
        self.PFE = nn.Sequential(
            nn.Conv2d(inp_channels//3, n_feat, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1)
        )

        modules_body = []
        
        modules_body.append(TGMMDRM(n_feat, n_MDRM, height, width, chan_factor, bias, groups=1))
        modules_body.append(TGMMDRM(n_feat, n_MDRM, height, width, chan_factor, bias, groups=2))
        modules_body.append(TGMMDRM(n_feat, n_MDRM, height, width, chan_factor, bias, groups=4))
        modules_body.append(TGMMDRM(n_feat, n_MDRM, height, width, chan_factor, bias, groups=4))

        self.body = nn.Sequential(*modules_body)
        
        # upsample
        self.conv_up1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        #self.conv_hr = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)   

    def forward(self, inp_img, atten):
        if self.scale == 2:
            feat_inp = pixel_unshuffle(inp_img, scale=2)
            feat_atten = pixel_unshuffle(atten, scale=2)
        elif self.scale == 1:
            feat_inp = pixel_unshuffle(inp_img, scale=4)
            feat_atten = pixel_unshuffle(atten, scale=4)
        else:
            feat_inp = inp_img
            feat_atten = atten
    
        shallow_feats_inp = self.conv_in(feat_inp)
        shallow_feats_atten = self.PFE(feat_atten)
        #st()
        shallow_feats = torch.cat((shallow_feats_inp, shallow_feats_atten), dim=1)
        deep_feats = self.body(shallow_feats)[:,:self.n_feat,:,:]
        #print(deep_feats.shape)
        if self.task == 'defocus_deblurring':
            deep_feats += shallow_feats
            deep_feats = self.lrelu(self.conv_up1(F.interpolate(deep_feats, scale_factor=2, mode='nearest')))
            deep_feats = self.lrelu(self.conv_up2(F.interpolate(deep_feats, scale_factor=2, mode='nearest')))
            out_img = self.conv_out(deep_feats)

        else:
            deep_feats = self.lrelu(self.conv_up1(F.interpolate(deep_feats, scale_factor=2, mode='nearest')))
            deep_feats = self.lrelu(self.conv_up2(F.interpolate(deep_feats, scale_factor=2, mode='nearest')))
            out_img = self.conv_out(deep_feats)
            
        return out_img
    
    
if __name__=='__main__':
    from ipdb import set_trace
    inp = torch.randn((1, 3, 128, 128))
    atten = torch.randn((1, 1, 128, 128))

    model= SimUESR(inp_channels=3, out_channels=3, n_feat=80, chan_factor=1.5, n_MDRM=2, height=3,
                          width=2, scale=2, bias=False, task=None)
    out = model(inp, atten)
    
    print(out.shape)
    # print(out[:,80:,:,:]==x_cat[:,80:,:,:])