import torch
import torch.nn as nn


"""
this layer is linear layer. inputs: input feature dimension, output feature dimension, bools for bias and batch norm.
"""
class LinearLayer(nn.Module):
    def __init__(self,
                 inputDim,
                 outputDim,
                 bias = True,
                 batchnorm = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.inputDim = inputDim
        self.outputDim = outputDim
        self.bias = bias
        self.batchnorm = batchnorm
        
        self.linear = nn.Linear(self.inputDim, self.outputDim, bias = self.bias and not self.batchnorm) # set linear layer
        if self.batchnorm: # if batch norm is used add batch norm
             self.batchnorm = nn.BatchNorm1d(self.outputDim)

    def forward(self,x):
        x = self.linear(x) #pass input through linear layer
        if self.batchnorm: #if batchnorm, apply bn
            x = self.bn(x)
        return x