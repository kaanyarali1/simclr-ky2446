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


"""
this class if for ResNet50-CIFAR10 instance level classification. Change the first conv and pooling layers to maintain the spatial size of the output.
"""
class ResNetCifar(nn.Module):
    def __init__(self,name,batchSize,momentum,T,K):
        super().__init__()
        
        self.encoder = get_resnet(name)
        
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False) #make 3*3 conv instaed of 7*7
        self.encoder.maxpool = Identity() #remove pooling layer
        
        self.encoder.fc = Identity()

        self.normalize = Normalize(2) #add l2 normalization layer
        
        for p in self.encoder.parameters():
            p.requires_grad = True

        self.unigrams = torch.ones(50000) #training set size 50k, create 50k unigram probs
        stdv = 1. / math.sqrt(2048/3) #specific radom init for memorybank
        self.memory = torch.rand((50000,2048)).mul(2*stdv).add(-stdv) #create memory bank
        self.multinomial = AliasMethod(self.unigrams) #sampler method
        self.multinomial.cuda() #load to cuda
        self.K = K # number of neg samples

        self.momentum = momentum # for proximal regularization, preventting noisy updates on memory bank
        self.batchSize = batchSize   #set batch size
        self.T = T #set temp

    def forward(self,x,y):
        out = self.encoder(x) #get feature embedding
        features = self.normalize(out) #normalize embedding

        idx = self.multinomial.draw(self.batchSize * (self.K)).view(self.batchSize,-1) #draw idx from memory banl
        idx.select(1,0).copy_(y.data) #add positive samples idx to first column

        idx = idx.cuda() #load cuda 
        self.memory = self.memory.cuda() #load cuda

        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight.resize_(self.batchSize, self.K, 2048) #batch size, number of neg samples, embedding dim

        with torch.no_grad():
          temp_features = features.reshape(self.batchSize, 2048, 1)
        output = torch.bmm(weight,temp_features) # calculate similarity
        output.div_(self.T).exp_() # divide by denom

        weight_pos = weight.select(1, 0).resize_((self.batchSize, 2048)) # select positive pairs from matrix
        weight_pos.mul_(self.momentum) #proximal reg momentum
        weight_pos.add_(torch.mul(features.data, 1-self.momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        y = y.cuda()
        temp = torch.clone(self.memory)
        temp.index_copy_(0, y, updated_weight)
        #self.memory.index_copy_(0, y, weight_pos)

        output = output / 52580 # divide by normalization constant
        output = output.reshape(self.batchSize,self.K)
        
        return output, temp #shape (batchsize,total no images(pos+neg imgs))

    def getEmbedding(self,x):
      out = self.encoder(x) #pass input images through network
      features = self.normalize(out) #normalize embedding
      return out #get embedding 
  
    def updateMemory(self,memory):
      self.memory = memory #update memory bank
