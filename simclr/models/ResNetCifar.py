import torch
import torch.nn as nn
import torchvision.models as models
from layers.identity import Identity
from layers.projection import ProjectionHead
from layers.Normalization import Normalize
from resnet import get_resnet


class ResNetCifar(nn.Module):
    def __init__(self,name):
        super().__init__()
        
        self.encoder = get_resnet(name)
        
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.encoder.maxpool = Identity()
        
        self.encoder.fc = Identity()

        self.normalize = Normalize(2)
        
        for p in self.encoder.parameters():
            p.requires_grad = True
        

    def forward(self,x):
        out = self.encoder(x)
        out = self.normalize(out)
        
        return out
