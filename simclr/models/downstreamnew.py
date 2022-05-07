import torch.nn as nn

class DownStreamNew(nn.Module):
    def __init__(self, encoder,proj, n_features, n_classes):
        super(DownStreamNew, self).__init__()

        self.encoder = encoder
        self.proj = proj
        self.model = nn.Linear(n_features, n_classes)

        self.encoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.proj(out)
        return self.model(out)