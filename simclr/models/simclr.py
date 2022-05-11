import torch
import torch.nn as nn
import torchvision.models as models
from layers.identity import Identity
from layers.projection import ProjectionHead
from resnet import get_resnet

"""
this class is for creating models for SimCLR. Since CIFAR10 and ResNet50 are used throughout the project, we modify the inital conv layer, and pooling layer to maintain the spatial size properly. 
"""
class SimClr(nn.Module):
    def __init__(self,name,projectedDimensionality):
        super().__init__()
        
        self.encoder = get_resnet(name) # get resnet 
        
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False) #change conv 7*7 to 3*3
        self.encoder.maxpool = Identity() #remove pooling layer and make it identity layer
        
        self.encoder.fc = Identity() #remove fc layer from resnet
        
        for p in self.encoder.parameters():
            p.requires_grad = True 
        
        self.projector = ProjectionHead(2048, 2048, projectedDimensionality) # projector will have input from encoder sized (2048). Do non linear projection
                                                              # 2048->2048->projected dimensionality
    def forward(self,x_i,x_j):
        h_i = self.encoder(x_i) # get embedding for the first pair
        h_j = self.encoder(x_j) # get embedding for the second pair
        
        z_i = self.projector(h_i) # project the first embedding
        z_j = self.projector(h_j) # project the second embedding
        
        return h_i,h_j,z_i,z_j # return embeddings and projections for both pairs
