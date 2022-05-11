import torch.nn as nn
import torch 
import torch.nn as nn
import numpy as np


"""
this is the implementation for NX-Tent loss function used in SimCLR paper.
"""

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        
        self.temperature = temperature # set temperature 
        self.batchSize = batch_size # set batch size

        self.mask = self.createMaskForPositiveSamples(batchSize) # create binary mask for possitive samples sized 2n*2n
        self.similarityFunction = nn.CosineSimilarity(dim=2) # define cosine similarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum") #set reduction to sum, for summing the softmax(logits).
    

    def createMaskForPositiveSamples(self, batch_size):
        total_imgs = 2 * batch_size # there will be 2N images in total
        mask = torch.ones((total_imgs, total_imgs), dtype=bool) # create mask for 2n*2n images.
        mask = mask.fill_diagonal_(0) # initialize diagonal values for 0. Diagonal values is similarites with itself. they are not used in loss function. 
        
        for i in range(batch_size): # set positive samples idx with zero in the matrix. for example there are 3 imgs before augmentation. there will 6 images after augmentation 
            mask[i, batch_size + i] = 0 #for ex, images index 0 and 3 are correlated positive samoples, set their bool mask value as 0
            mask[batch_size + i, i] = 0 # also do for lower triangle in the matrix
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size #total images become 2n
        z = torch.cat((z_i, z_j), dim=0) # concatenate projected embeddings
        sim = self.similarityFunction(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature # calculate similarity function and divied by temperature
        upperSim = torch.diag(sim, self.batch_size) # get upper triangle similarities
        lowerSim = torch.diag(sim, -self.batch_size) # get lower triangle similarities
        positiveSamples = torch.cat((upperSim, lowerSim), dim=0).reshape(N, 1) # get correlated/positive samples prob. reshape to (n,1)
        negativeSamples = sim[self.mask].reshape(N, -1) #using the mask we generated, get negative samples. reshape to (n,1)
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positiveSamples.device).long() # set auxilary labels sized N, having all values 0 for softmax sum.
        logits = torch.cat((positive_samples, negative_samples), dim=1) # concatenate positive and negative samples for softmax sum
        loss = self.criterion(logits, labels) # apply softmax sum 
        loss /= N # divide by number of samples in the mini-batch
        
        return loss