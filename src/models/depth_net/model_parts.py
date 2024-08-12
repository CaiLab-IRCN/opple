""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F

negative_slope = 0.1
dropout = torch.nn.Dropout(0.05)

class Down(nn.Module):
    """
    convolution => Leaky ReLU => [BN] => Dropout =>
        Down convolution (stride = 2) => leaky ReLU => [BN] => Dropout

    Normalization is optional
    Order of Leaky relu and Norm was different for every implementation; relu -> BN -> dropout
    worked better for us
    We also tried Instance norm but it was not as good
    No point of using maxpooling in this method, downsampling is done by increasing the stride
    """
    def __init__(self, in_channels, out_channels, input_size,\
            norm=True, mid_channels=None, eps=1e-4):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels
        size = [s//2 for s in input_size]
        conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(conv.weight, a=negative_slope)
        down = nn.Conv2d(mid_channels, out_channels, stride=2, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(down.weight, a=negative_slope)
        
        if norm:
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.BatchNorm2d(mid_channels, eps=eps),
                dropout
            )
            self.down = nn.Sequential(
                down,
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.BatchNorm2d(out_channels, eps=eps),
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
    """Upscaling then double conv"""
    """
    Up convolution (stride=2) => Leaky ReLU => [BN] => Dropout =>
        convolution => leaky ReLU => [BN] => Dropout

    Normalization is optional
    Order of Leaky relu and Norm was different for every implementation; relu -> BN -> dropout
    worked better for us
    We also tries Instance norm and Layernorm, but it was not as good
    No point of using maxpooling in this method, downsampling is done by increasing the stride
    """

    def __init__(self, in_channels, out_channels,\
            input_size, norm=True, mid_channels=None,\
            eps=0.01): # bilinear=True, 
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        size = [s*2 for s in input_size]
        up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2, padding=0)
        torch.nn.init.kaiming_uniform_(up.weight, a=negative_slope)
        if norm:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.BatchNorm2d(mid_channels, eps=eps),
                dropout
            )
        else:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )
        conv = nn.Conv2d(mid_channels + out_channels, out_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_uniform_(conv.weight, a=negative_slope)
        if norm:
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.BatchNorm2d(out_channels, eps=eps),
                dropout
            )
        else:
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope),
                dropout
            )

    def forward(self, x1, x2):
        """
        x1: input to the layer/output of the previous layer
        x2: input from the skip connection
        """
        x1_up = self.up(x1)
        # input is CHW
        diffY = x1_up.size()[2] - x2.size()[2]
        diffX = x1_up.size()[3] - x2.size()[3]
        x1_up = x1_up[:,:,diffY // 2:x1_up.size()[2]-(diffY - diffY // 2),\
                diffX // 2:x1_up.size()[3]-(diffX - diffX // 2)] 

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1_up], dim=1)
        return self.conv(x) + x1_up


class OutConv(nn.Module):
    """
    Convolution function definition for convolving last layer to output channels
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class InConv(nn.Module):
    """
    Convolution function definition for convolving input channels to defined channels
    """
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
