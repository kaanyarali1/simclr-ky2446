import torch
import numpy as np

class AliasMethod(object):
    '''
        Inspired From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        but we updated...
        this class is used for sampling negative datapoints from memory bank. 
    '''
    def __init__(self, probabilities):

    
        K = len(probabilities)
        self.alias = torch.LongTensor([0]*K) #creates empty tensor shape len(probs) having all values zero
        self.probabilities = torch.zeros(K) #create empty tensor

        if probabilities.sum() > 1: #if prob sum bigger than 1, normalize it.
            probabilities.div_(probabilities.sum())
        

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        small_part = [] # create empty list for probs smaller than 1/K
        large_part = [] # create empty list for probs bigger than 1/K
        for idx, prob in enumerate(probabilities):
            self.probabilities[idx] = K*prob
            if self.probabilities[idx] < 1.0: #check whether prob is smaller than 1/K
                small_part.append(idx) #if smaller, append
            else:
                large_part.append(idx) #if bigger, append


        number_of_smaller = len(small_part) #total number of smaller probs
        number_of_bigger = len(large_part)  #total number of bigger probs


        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while number_of_smaller > 0 and number_of_bigger > 0:
            small = small_part.pop()
            large = large_part.pop()

            self.alias[small] = large
            self.probabilities[large] = (self.probabilities[large] - 1.0) + self.probabilities[small]

            if self.probabilities[large] < 1.0:
                small_part.append(large)
            else:
                large_part.append(large)

        for last_one in small_part+large_part:
            self.probabilities[last_one] = 1

    def cuda(self): 
        self.prob = self.prob.cuda() #load to GPU
        self.alias = self.alias.cuda() # load to GPU

    def draw(self, N):
        '''
            Draw N samples from multinomial
        '''
        K = self.alias.size(0) 

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.probabilities.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj

