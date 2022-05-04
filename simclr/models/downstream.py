import torch.nn as nn


class DownStream(nn.Module):
    def __init__(self, encoder, n_features, n_classes):
        super(DownStream, self).__init__()

        self.encoder = encoder
        self.model = nn.Linear(n_features, n_classes)

        self.encoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.encoder(x)
        return self.model(out)