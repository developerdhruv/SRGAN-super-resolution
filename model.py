import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    #conv- bn-relu block
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            discriminator: bool = False,
            use_act = True,
            use_bn = True,
            **kwargs

            
            ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2) if discriminator else nn.PReLU(num_parameters=out_channels)
    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
        

class UpsamebleBlock(nn.Module):
    def __init__(self, in_channels: int, scale_fator:int = 2):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_fator ** 2, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(scale_fator)
        self.act  = nn.PReLU(num_parameters=in_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3,stride = 1, padding=1, )
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3,stride = 1, padding=1, use_act=False)
    def forward(self,x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        return x + x2






class Generator(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_residual_blocks:int = 16):
        super().__init__()
        self.initial = ConvBlock(in_channels, 64, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]

        )
        self.convblock = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, use_act= False)
        self.upsample1 = UpsamebleBlock(64, scale_fator=2)
        self.upsample2 = UpsamebleBlock(64, scale_fator=2)
        self.final = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
    def forward(self, x):
        first = self.initial(x)
        x = self.residuals(first)
        x = self.convblock(x) + first
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.final(x)
        return torch.tanh(x)







class Discriminator(nn.Module):
    def __init__(self,in_channels = 3 , features = [64,64, 128, 256, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(in_channels, feature, kernel_size=3, stride=1 if idx == 0 else 2, padding=1, use_bn=False if idx == 0 else True, use_act= True, discriminator=True)
            )
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)
       # self.final = nn.Conv2d(features[-1], 1, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512* 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace =True),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        x= self.blocks(x)
        return  self.classifier(x)
          