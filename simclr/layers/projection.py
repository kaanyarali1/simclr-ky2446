import torch
import torch.nn as nn
from layers.linear import LinearLayer


"""
this layer is used for projecting the input embedding to another space either using linear layer or non linear layer.
inputs: input embedding dimensionality, output embedding dimensionality, hidden embedding dimensionality, layer type
"""
class ProjectionHead(nn.Module):
    def __init__(self,
                 inputDim,
                 hiddenDim,
                 outputDim,
                 layerType = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.inputDim = inputDim  # set input embedding dimension
        self.outputDim = outputDim # set output embedding dimension
        self.hiddenDim = hiddenDim # set hidden embedding dimension
        self.layerType = layerType # set layer type

        if self.layerType == 'linear': # if linear, do not add any non-linear linearity
            self.layers = LinearLayer(self.inputDim,self.outputDim,False, True) # this is never used, written for convenience  
        elif self.layerType == 'nonlinear': # if non linear, project first to hidden state dim. apply non-linearity, then project to output embedding dimensionality
            self.layers = nn.Sequential(
                LinearLayer(self.inputDim,self.hiddenDim,False, False), # do not use bn or bias
                nn.ReLU(),
                LinearLayer(self.hiddenDim,self.outputDim,False,False)) # do not use bn or bias
        
    def forward(self,x):
        x = self.layers(x) # pass the inpu through projection layer
        return x

