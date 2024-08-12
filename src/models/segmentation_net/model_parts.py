""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F

negative_slope = 0.1
dropout = torch.nn.Dropout(0.05)

class Down(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, input_size, norm=True, mid_channels=None, eps=1e-6):
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels
        size = [s//2 for s in input_size]
        conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(conv.weight, a=negative_slope)
        down = nn.Conv2d(mid_channels, out_channels, stride=2, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(down.weight, a=negative_slope)
        
        if norm:
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.BatchNorm2d(mid_channels, eps=eps),
                dropout
            )
            self.down = nn.Sequential(
                down,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.BatchNorm2d(out_channels, eps=eps),
                dropout
            )
        else:
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                dropout
            )
            self.down = nn.Sequential(
                down,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                dropout
            )

    def forward(self, x):
        mid = self.conv(x)
        out = self.down(mid + x)
        return out

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

####  The Atrous Spatial Pyramid Pooling

def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.BatchNorm2d(out_channles))

class ASSP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, out_channels, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, out_channels, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, out_channels, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, out_channels, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

        self.conv1 = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.dropout = dropout # nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), mode='bilinear', size=(x.size(2), x.size(3)),
                           align_corners=True) # 
        
        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

class Up(nn.Module):
    """Upscaling then single conv"""

    def __init__(self, in_channels, out_channels, input_size, bilinear=True, norm=True, mid_channels=None, eps=0.01): 
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        size = [s*2 for s in input_size]
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = nn.Conv2d(in_channels + mid_channels, out_channels, kernel_size=3, padding=1)
            normup = nn.BatchNorm2d(in_channels, eps=eps)
            norm = nn.BatchNorm2d(out_channels, eps=eps)
            self.repeatc = out_channels//in_channels
        else:
            up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1,
                    output_padding=1)
            conv = nn.Conv2d(mid_channels + out_channels, out_channels, kernel_size=3, padding=1)
            normup = nn.BatchNorm2d(mid_channels, eps=eps)
            norm = nn.BatchNorm2d(out_channels, eps=eps)
            torch.nn.init.kaiming_normal_(up.weight, a=negative_slope)
            self.repeatc = out_channels//mid_channels
        torch.nn.init.kaiming_normal_(conv.weight, a=negative_slope)
        if norm:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                normup,
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                norm,
                dropout
            )
        else:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
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
        return self.conv(x) # + x1_up 

class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, input_size, bilinear=True, norm=True, mid_channels=None, eps=0.01):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        size = [s*2 for s in input_size]
        if bilinear:
            up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            normup = nn.BatchNorm2d(in_channels, eps=eps)
            norm = nn.BatchNorm2d(out_channels, eps=eps)
        else:
            up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1,
                    output_padding=1)
            conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
            normup = nn.BatchNorm2d(mid_channels, eps=eps)
            norm = nn.BatchNorm2d(out_channels, eps=eps)
            torch.nn.init.kaiming_normal_(up.weight, a=negative_slope)
        torch.nn.init.kaiming_normal_(conv.weight, a=negative_slope)
        if norm:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                normup,
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                norm,
                dropout
            )
        else:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                dropout
            )

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        # input is CHW
        if(x2 is None):
            return self.conv(x1_up) # + x1_up
        else:
            print("use UP1 class for skip connections")

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
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
            normup = nn.BatchNorm2d(in_channels, eps=eps)
            norm = nn.BatchNorm2d(out_channels, eps=eps)
        else:
            up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1,
                    output_padding=1)
            conv = nn.Conv2d(mid_channels+16, out_channels, kernel_size=3, padding=1)
            normup = nn.BatchNorm2d(mid_channels, eps=eps)
            norm = nn.BatchNorm2d(out_channels, eps=eps)
            torch.nn.init.kaiming_normal_(up.weight, a=negative_slope)
        torch.nn.init.kaiming_normal_(conv.weight, a=negative_slope)
        if norm:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                normup,
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                norm,
                dropout
            )
        else:
            self.up = nn.Sequential(
                up,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                dropout
            )
            self.conv = nn.Sequential(
                conv,
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
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
            return x #+ x1
            

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')

    def forward(self, x):
        return self.conv(x)
    
class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(conv.weight, a=negative_slope)
        self.conv = nn.Sequential(
            conv,
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
