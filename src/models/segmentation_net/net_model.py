""" Full assembly of the parts to form the complete network """

import numpy as np
import torch.nn.functional as F

from .model_parts import *

negative_slope = 0.1
class SegNet(nn.Module):
    def __init__(self, cfg, params):
        super(SegNet, self).__init__()
        self.ils = cfg.SEGNET.intermediate_latent_split
        self.useRNN = cfg.SEGNET.use_RNN
        self.n_slots = cfg.TRAIN.num_slots
        self.skip_in_higher = cfg.SEGNET.skip_in_higher

        norm = cfg.SEGNET.use_norm
        num_bins = cfg.ROTATION.num_bins
        latent_midlayer = cfg.SEGNET.latent_midlayer
        bilinear = cfg.SEGNET.bilinear
        denomin = cfg.SEGNET.denomin
        n_classes = cfg.SEGNET.n_classes
        n_channels = cfg.SEGNET.n_channels
        output_latent_size =  self.n_slots * (3 + num_bins + cfg.TRAIN.segment_id_size)

        self.inc = InConv(n_channels, 512//denomin)
        size = cfg.SEGNET.input_size
        self.down1 = Down(512//denomin, 512//denomin, size, norm=norm)
        size = [int(np.ceil(s/2)) for s in size]
        self.down2 = Down(512//denomin, 512//denomin, size, norm=norm)
        size = [int(np.ceil(s/2)) for s in size]
        self.down3 = Down(512//denomin, 1024//denomin, size, norm=norm)
        size = [int(np.ceil(s/2)) for s in size]
        self.assp = ASSP(1024//denomin, 1024//denomin, 16) ### output stride of 16 for input size of 128
        self.down4 = Down(1024//denomin, 1024//denomin, size, norm=norm)
        size = [int(np.ceil(s/2)) for s in size]
        self.down5 = Down(1024//denomin, 2048//denomin, size, norm=norm)
        size = [int(np.ceil(s/2)) for s in size]
        self.down6 = Down(2048//denomin, 2048//denomin, size, norm=norm)
        size = [int(np.ceil(s/2)) for s in size]
        self.down7 = Down(2048//denomin, 2048//denomin, size, norm=norm)
        size = [int(np.ceil(s/2)) for s in size]
        input_length = int(np.prod(size) * 2048//denomin)
        
        ## code for splitting the latent code in depth and segmentation explicitly
        self.ils_depth = int(input_length*self.ils)
        self.ils_att = input_length - self.ils_depth

        self.up1_att_lin = nn.Sequential(torch.nn.Linear(output_latent_size//self.n_slots -3 -num_bins,\
                input_length), nn.LeakyReLU(negative_slope=negative_slope, inplace=False))

        self.up0_att = Up(2048//denomin, 2048//denomin, size, mid_channels = 2048//denomin,
                bilinear = bilinear, norm=norm)
        size = [s*2 for s in size]
        self.up1_att = Up(2048//denomin, 2048//denomin, size, mid_channels = 2048//denomin,
                bilinear = bilinear, norm=norm)
        size = [s*2 for s in size]
        self.up2_att = Up(2048//denomin, 1024//denomin, size, mid_channels = 1024//denomin,
                bilinear = bilinear, norm=norm)
        size = [s*2 for s in size]

        if not self.skip_in_higher:
            self.up3_att = Up2(1024//denomin, 1024//denomin, size, mid_channels = 1024//denomin,
                    bilinear = bilinear, norm=norm)
            size = [s*2 for s in size]
            self.up4_att = Up2(1024//denomin, 512//denomin, size, mid_channels = 512//denomin, bilinear
                    = bilinear, norm=norm)
            size = [s*2 for s in size]
            self.up5_att = Up2(512//denomin, 512//denomin, size, mid_channels = 512//denomin, bilinear =
                    bilinear, norm=norm)
            size = [s*2 for s in size]
            self.up6_att = Up2(512//denomin, 512//denomin, size, mid_channels = 512//denomin, bilinear =
                    bilinear, norm=norm)
        else:
            self.up3_att = Up(1024//denomin, 1024//denomin, size, mid_channels = 1024//denomin,
                    bilinear = bilinear, norm=norm)
            size = [s*2 for s in size]
            self.up4_att = Up(1024//denomin, 512//denomin, size, mid_channels = 512//denomin, bilinear
                    = bilinear, norm=norm)
            size = [s*2 for s in size]
            self.up5_att = Up(512//denomin, 512//denomin, size, mid_channels = 512//denomin, bilinear =
                    bilinear, norm=norm)
            size = [s*2 for s in size]
            self.up6_att = Up(512//denomin, 512//denomin, size, mid_channels = 512//denomin, bilinear =
                    bilinear, norm=norm)
        self.outc_att = OutConv(512//denomin, n_classes) ## nclasses = num maps 

        if not self.useRNN:
            
            if latent_midlayer is None:
                latent_midlayer = input_length//2
            self.lin1 = nn.Sequential(torch.nn.Linear(input_length, latent_midlayer),
                                      nn.LeakyReLU(negative_slope=negative_slope, inplace=False))

            self.lin2 = nn.Sequential(torch.nn.Linear(latent_midlayer, output_latent_size))
        else:
            if latent_midlayer is None:
                latent_midlayer = input_length

            self.rnn = nn.LSTM(input_length, latent_midlayer, 1, batch_first=True)

            nn.init.kaiming_normal_(self.rnn.weight_ih_l0)
            nn.init.kaiming_normal_(self.rnn.weight_hh_l0)

            self.lin = nn.Sequential(nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
                         torch.nn.Linear(latent_midlayer, output_latent_size//self.n_slots))
        
        self.outp = torch.nn.Sigmoid()

    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.assp(x4)
        x5 = self.down4(x5)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)

        x9 = x8.flatten(1,-1)
        links = [x1, x2, x3, x4, x5, x6, x7, x8]
        return x8, x9, links

    def decoder_attention(self, x, links, shape):
        # 'x' shape:        [batch x latent_id_size] ie, [20,64] (from latent_transformed)
        x1, x2, x3, x4, x5, x6, x7, x8 = links

        ## lineraly interpolating the object codes up to x7 shape
        x_out = self.up1_att_lin(x)
        x_out = torch.reshape(x_out, shape)

        ## using the same decoder as depth output
        x_out = self.up0_att(x_out, x7)
        x_out = self.up1_att(x_out, x6)
        x_out = self.up2_att(x_out, x5)
        if self.skip_in_higher:

            x_out = self.up3_att(x_out, x4)
            x_out = self.up4_att(x_out, x3)
            x_out = self.up5_att(x_out, x2)
            x_out = self.up6_att(x_out, x1)
        else:
            x_out = self.up3_att(x_out, None)
            x_out = self.up4_att(x_out, None)
            x_out = self.up5_att(x_out, None)
            x_out = self.up6_att(x_out, None)
        logits = self.outc_att(x_out)
        # 'logits' shape =  [batch x 1 x height x width] ie: [20,1,128,128]
        return logits

    def latent_out(self, x):
        if self.useRNN:
            x1_latent = self.rnn(x[:,None,:].expand(-1, self.n_slots, -1))
            x_latent = self.lin(x1_latent[0])
        else:
            x1_latent = self.lin1(x.flatten(1,-1))
            x_latent = self.lin2(x1_latent)
        return x_latent
