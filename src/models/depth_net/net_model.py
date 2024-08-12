""" Assembly of the parts to form the complete depth network """

import torch.nn.functional as F
from .model_parts import *

class DepthNet(nn.Module):
    """
        Depth inference network model 
        We use UNet as main backbone for our network

                        |-------------------------------------------|>
        Input -----> {Conv} x6 ----> Low Dim Representation ----> {Deconv}x6 -----> Depth

        n_channels: number of input channels to define the input layer
        n_classes: number of channels in the output; for depth this would be 1
        bilinear: toggle between bilinear and gaussian interpolation
        eps: epsilon parameter for the batchnorm2d -- numerical stability value

        TODO: Check run with lower eps value
    """
    def __init__(self, cfg, params):
        super(DepthNet, self).__init__()
        self.n_channels = cfg.DEPTHNET.n_channels
        self.n_classes = cfg.DEPTHNET.n_classes
        self.bilinear = cfg.DEPTHNET.bilinear
        self.eps = cfg.DEPTHNET.eps

        denomin = 4
        self.inc = InConv(self.n_channels, 64//denomin)
        size = cfg.DEPTHNET.input_size
        self.down1 = Down(64//denomin, 128//denomin, size, norm=True, eps=self.eps)
        size = [s//2 for s in size]
        self.down2 = Down(128//denomin, 256//denomin, size, norm=True, eps=self.eps)
        size = [s//2 for s in size]
        self.down3 = Down(256//denomin, 512//denomin, size, norm=True, eps=self.eps)
        size = [s//2 for s in size]
        self.down4 = Down(512//denomin, 1024//denomin, size, norm=True, eps=self.eps)
        size = [s//2 for s in size]
        self.down5 = Down(1024//denomin, 1024//denomin, size, norm=True, eps=self.eps)
        size = [s//2 for s in size]
        self.down6 = Down(1024//denomin, 1024//denomin, size, norm=False, eps=self.eps)

        size = [s//2 for s in size]
        self.up1 = Up(1024//denomin, 1024//denomin, size, norm=False, eps=self.eps)
        size = [s*2 for s in size]
        self.up2 = Up(1024//denomin, 1024//denomin, size, norm=True, eps=self.eps)
        size = [s*2 for s in size]
        self.up3 = Up(1024//denomin, 512//denomin, size, norm=True, eps=self.eps)
        size = [s*2 for s in size]
        self.up4 = Up(512//denomin, 256//denomin, size, norm=True, eps=self.eps)
        size = [s*2 for s in size]
        self.up5 = Up(256//denomin, 128//denomin, size, norm=True, eps=self.eps)
        size = [s*2 for s in size]
        self.up6 = Up(128//denomin, 64//denomin, size, norm=True, eps=self.eps)
        self.outc = OutConv(64//denomin, self.n_classes)

    def forward(self, x):
        # downconv
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        #upconv
        x_out = self.up1(x7, x6)
        x_out = self.up2(x_out, x5)
        x_out = self.up3(x_out, x4)
        x_out = self.up4(x_out, x3)
        x_out = self.up5(x_out, x2)
        x_out = self.up6(x_out, x1)
        logits = self.outc(x_out)
        return logits
