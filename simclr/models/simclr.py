import torch
import torch.nn as nn
import torchvision.models as models
from layers.identity import Identity
from layers.projection import ProjectionHead
from resnet import get_resnet


class SimClr(nn.Module):
    def __init__(self,name):
        super().__init__()
        
        self.encoder = get_resnet(name)
        
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.encoder.maxpool = Identity()
        
        self.encoder.fc = Identity()
        
        for p in self.encoder.parameters():
            p.requires_grad = True
        
        self.projector = ProjectionHead(2048, 2048, 128)

    def forward(self,x_i,x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return h_i,h_j,z_i,z_j
