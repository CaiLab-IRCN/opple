""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



negative_slope = 0.1
dropout = torch.nn.Dropout(0.0005)

class Down(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, input_size, norm=True, mid_channels=None, eps=0.01):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels
        size = [int(np.ceil(s/2)) for s in input_size]
        conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(conv.weight, a=negative_slope)
        down = nn.Conv2d(mid_channels, out_channels, stride=2, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(down.weight, a=negative_slope)
        
        if norm:
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.LayerNorm([mid_channels] + input_size, eps=eps, elementwise_affine=False),
                dropout
            )
            self.down = nn.Sequential(
                down,
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.LayerNorm([out_channels] + size, eps=eps, elementwise_affine=False),
                dropout
            )
        else:
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )
            self.down = nn.Sequential(
                down,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )

    def forward(self, x):
        mid = self.conv(x)
        out = self.down(mid + x)
        return out


class Up(nn.Module):
    """Upscaling then single conv"""

    def __init__(self, in_channels, out_channels, input_size, output_size, bilinear=True, norm=True, mid_channels=None, eps=0.01):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        size = [s*2 for s in input_size]
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = nn.Conv2d(in_channels + mid_channels, out_channels, kernel_size=3, padding=1)
            normup = nn.LayerNorm([in_channels] + size, eps=eps, elementwise_affine=False)
            norm = nn.LayerNorm([out_channels] + output_size, eps=eps, elementwise_affine=False)
        else:
            up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1,
                    output_padding=1)
            conv = nn.Conv2d(mid_channels + out_channels, out_channels, kernel_size=3, padding=1)
            normup = nn.LayerNorm([mid_channels] + size, eps=eps, elementwise_affine=False)
            norm = nn.LayerNorm([out_channels] + output_size, eps=eps, elementwise_affine=False)
            torch.nn.init.kaiming_uniform_(up.weight, a=negative_slope)
        torch.nn.init.kaiming_uniform_(conv.weight, a=negative_slope)
        if norm:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope),
                normup,
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                norm,
                dropout
            )
        else:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        # input is CHW
        if(x2 is None):
            print("x2 is none, use up2() for this function")

        diffY = x1_up.size()[2] - x2.size()[2]
        diffX = x1_up.size()[3] - x2.size()[3]
        x1_up = x1_up[:,:,diffY // 2:x1_up.size()[2]-(diffY - diffY // 2),diffX // 2:x1_up.size()[3]-(diffX - diffX // 2)] 

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x)

class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, input_size, output_size, bilinear=True, norm=True, mid_channels=None, eps=0.01):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        size = [s*2 for s in input_size]
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            normup = nn.LayerNorm([in_channels] + size, eps=eps, elementwise_affine=False)
            norm = nn.LayerNorm([out_channels] + output_size, eps=eps, elementwise_affine=False)
        else:
            up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1,
                    output_padding=1)
            conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
            normup = nn.LayerNorm([mid_channels] + size, eps=eps, elementwise_affine=False)
            norm = nn.LayerNorm([out_channels] + output_size, eps=eps, elementwise_affine=False)
            torch.nn.init.kaiming_uniform_(up.weight, a=negative_slope)
        torch.nn.init.kaiming_uniform_(conv.weight, a=negative_slope)
        if norm:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope),
                normup,
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                norm,
                dropout
            )
        else:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        # input is CHW
        if(x2 is None):
            return self.conv(x1_up)
        else:
            print("use UP1 class for skip connections")

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
            
class Up3(nn.Module):
    """Upscaling with first 16 channels only then conv"""

    def __init__(self, in_channels, out_channels, input_size, bilinear=True,  norm=True, mid_channels=None, eps=0.01):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        size = [s*2 for s in input_size]
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = nn.Conv2d(in_channels+16 , out_channels, kernel_size=3, padding=1)
            normup = nn.LayerNorm([in_channels] + size, eps=eps, elementwise_affine=False)
            norm = nn.LayerNorm([out_channels] + size, eps=eps, elementwise_affine=False)
        else:
            up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1)
            conv = nn.Conv2d(mid_channels+16, out_channels, kernel_size=3, padding=1)
            normup = nn.LayerNorm([mid_channels] + size, eps=eps, elementwise_affine=False)
            norm = nn.LayerNorm([out_channels] + size, eps=eps, elementwise_affine=False)
            torch.nn.init.kaiming_uniform_(up.weight, a=negative_slope)
        torch.nn.init.kaiming_uniform_(conv.weight, a=negative_slope)
        if norm:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope),
                normup,
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                norm,
                dropout
            )
        else:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        if(x2 is None):
            print("use UP2 class for no skip connections")
        else:
            x2_16 = x2[:,:16] ## taking first 16 channels
            x = torch.cat([x2_16, x1_up], dim=1)
            x = self.conv(x)
            return x
            

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(conv.weight, a=negative_slope)
        self.conv = nn.Sequential(
            conv,
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, x):
        return self.conv(x)
