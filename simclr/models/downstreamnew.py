import torch.nn as nn


""" 
this class if for creating models for downstream task. it uses the encoder network from the simclr training.
"""
class DownStreamNew(nn.Module):
    def __init__(self, encoder,proj, n_features, n_classes):
        super(DownStreamNew, self).__init__()

        self.encoder = encoder # set encoder network
        self.proj = proj #set projector network
        self.model = nn.Linear(n_features, n_classes) #final classification layer

        self.encoder.eval() # set encoder in evaluation mode

        for p in self.encoder.parameters():
            p.requires_grad = False # freeze encoder weights
        for p in self.proj.parameters():
            p.requires_grad = False # freeze projector weights
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.proj(out)
        return self.model(out) 