import torch
import torch.nn as nn
import torchvision.models as models
from layers.identity import Identity
from layers.projection import ProjectionHead
from layers.Normalization import Normalize
from layers.linear import LinearLayer
from resnet import get_resnet
from alias_multinomial import AliasMethod
import math


class ResNetCifar(nn.Module):
    def __init__(self,name,batchSize,momentum,T,K):
        super().__init__()
        
        self.encoder = get_resnet(name)
        
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.encoder.maxpool = Identity()
        
        self.encoder.fc = Identity()

        self.normalize = Normalize(2)
        
        for p in self.encoder.parameters():
            p.requires_grad = True

        self.unigrams = torch.ones(50000)
        stdv = 1. / math.sqrt(2048/3)
        self.memory = torch.rand((50000,2048)).mul(2*stdv).add(-stdv)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.momentum = momentum
        self.batchSize = batchSize      
        self.T = T

    def forward(self,x,y):
        out = self.encoder(x)
        features = self.normalize(out)

        idx = self.multinomial.draw(self.batchSize * (self.K)).view(self.batchSize,-1)
        idx.select(1,0).copy_(y.data)

        idx = idx.cuda()
        self.memory = self.memory.cuda()

        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight.resize_(self.batchSize, self.K, 2048)

        with torch.no_grad():
          temp_features = features.reshape(self.batchSize, 2048, 1)
        output = torch.bmm(weight,temp_features)
        output.div_(self.T).exp_() # batchSize * self.K+

        weight_pos = weight.select(1, 0).resize_((self.batchSize, 2048))
        weight_pos.mul_(self.momentum)
        weight_pos.add_(torch.mul(features.data, 1-self.momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        y = y.cuda()
        temp = torch.clone(self.memory)
        temp.index_copy_(0, y, updated_weight)
        #self.memory.index_copy_(0, y, weight_pos)

        output = output / 52580
        output = output.reshape(self.batchSize,self.K)
        
        return output, temp

    def getEmbedding(self,x):
      out = self.encoder(x)
      features = self.normalize(out)
      return out
  
    def updateMemory(self,memory):
      self.memory = memory
