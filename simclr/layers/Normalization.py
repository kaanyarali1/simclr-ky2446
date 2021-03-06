import torch
from torch import nn

"""
this layer makes L2 normalization to the input embedding.
"""
class Normalize(nn.Module):

    def __init__(self, lnorm=2):
        super(Normalize, self).__init__()
        self.lnorm = lnorm
    
    def forward(self, x):
        norm = x.norm(dim=1, p=self.lnorm,keepdim=True)
        out = x / (norm)
        return out