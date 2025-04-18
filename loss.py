import torch.nn as nn
from torchvision.models import vgg19
import config


class VGG19Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()
        for param in self.vgg.parameters():
            param.requires_grad = False
    def forward(self,input, target):
        input = self.vgg(input)
        target = self.vgg(target)
        loss = self.loss(input, target)
        return loss