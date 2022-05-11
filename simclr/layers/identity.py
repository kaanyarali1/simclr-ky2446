import torch
import torch.nn as nn

"""
this layer is identity layer, there is no computation in this layer.
"""
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x 