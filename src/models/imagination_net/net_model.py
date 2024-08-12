""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import numpy as np

from .model_parts import *

class ImagNet(nn.Module):
    def __init__(self, cfg, params):
        super(ImagNet, self).__init__()
        bilinear = cfg.IMAGNET.bilinear
        self.eps = cfg.IMAGNET.eps
        self.skip_in_higher = cfg.IMAGNET.skip_in_higher
        denomin = cfg.IMAGNET.denomin

        self.inc = InConv(cfg.IMAGNET.n_channels, 64//denomin)
        input_size = cfg.DATASET.frame_size
        size = input_size
        linmul = (input_size[0]//64)**2
        self.down1 = Down(64//denomin, 128//denomin, size, norm=True, eps=self.eps)
        size1 = [int(np.ceil(s/2)) for s in size]
        self.down2 = Down(128//denomin, 256//denomin, size1, norm=True, eps=self.eps)
        size2 = [int(np.ceil(s/2)) for s in size1]
        self.down3 = Down(256//denomin, 512//denomin, size2, norm=True, eps=self.eps)
        size3 = [int(np.ceil(s/2)) for s in size2]
        self.down4 = Down(512//denomin, 1024//denomin, size3, norm=True, eps=self.eps)
        size4 = [int(np.ceil(s/2)) for s in size3]
        self.down5 = Down(1024//denomin, 2048//denomin, size4, norm=True, eps=self.eps)
        size5 = [int(np.ceil(s/2)) for s in size4]
        # factor = 2 if bilinear else 1
        self.down6 = Down(2048//denomin, 4096//denomin, size5, norm=False, eps=self.eps)
        size6 = [int(np.ceil(s/2)) for s in size5]

        
        self.linear = torch.nn.Linear(cfg.IMAGINATION.imagination_params, np.prod(size6)*4096//denomin)
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        self.up1 = Up(4096//denomin, 2048//denomin, size6, size5, mid_channels = 2048//denomin,\
                norm=False, eps=self.eps, bilinear=bilinear)
        self.up2 = Up(2048//denomin, 1024//denomin, size5, size4, mid_channels = 1024//denomin,\
                norm=True, eps=self.eps, bilinear=bilinear)
        self.up3 = Up(1024//denomin, 512//denomin, size4, size3, mid_channels = 512//denomin,\
                norm=True, eps=self.eps, bilinear=bilinear)

        if self.skip_in_higher:
            self.up4 = Up2(512//denomin, 256//denomin, size3, size2, mid_channels = 256//denomin,\
                    norm=True, eps=self.eps, bilinear=bilinear)
            self.up5 = Up2(256//denomin, 128//denomin, size2, size1, mid_channels = 128//denomin,\
                    norm=True, eps=self.eps, bilinear=bilinear)
            self.up6 = Up2(128//denomin, 64//denomin, size1, size, mid_channels = 64//denomin,\
                    norm=True, eps=self.eps, bilinear=bilinear)
        else:
            self.up4 = Up(512//denomin, 256//denomin, size3, size2, mid_channels = 256//denomin,\
                    norm=True, eps=self.eps, bilinear=bilinear)
            self.up5 = Up(256//denomin, 128//denomin, size2, size1, mid_channels = 128//denomin,\
                    norm=True, eps=self.eps, bilinear=bilinear)
            self.up6 = Up(128//denomin, 64//denomin, size1, size, mid_channels = 64//denomin,\
                    norm=True, eps=self.eps, bilinear=bilinear)
        self.outc = OutConv(64//denomin, cfg.IMAGNET.n_classes)

    def forward(self, data_batch):
        x = data_batch['frame1']
        motion_params = data_batch['motion_params']

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = torch.flatten(x7, 1, -1)
        x10 = x8+self.linear(motion_params)
        x10 = x10.reshape(x7.shape)
        x = self.up1(x10, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        if not self.skip_in_higher:
            x = self.up4(x, x3)
            x = self.up5(x, x2)
            x = self.up6(x, x1)
        else:
            x = self.up4(x, None)
            x = self.up5(x, None)
            x = self.up6(x, None)
        logits = self.outc(x)
        return logits, x10
