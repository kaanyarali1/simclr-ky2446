import torch
from torch import nn

eps = 1e-7
class NCECriterion(nn.Module):

    def __init__(self, indexSize):
        super(NCECriterion, self).__init__()
        self.indexSize = indexSize

    def forward(self, x, targets):
        
        positive_denom_pre = 1 / float(self.indexSize) # calculate for the denominator part for nonparametric softmax
        negative_denom_pre = 1 / float(self.indexSize) # calculate for the denominator part for nonparametric softmax

        batch_size = x.size(0)
        K = x.size(1)-1
        
        probs_positive  = x.select(1,0) # select probabilities for positive samples 
        probs_positive_denom = probs_positive.add(eps + K * positive_denom_pre)
        final_positive_probs = torch.div(probs_positive, probs_positive_denom) # positive pair probabilities
        
        
        probs_negative_denom = x.narrow(1,1,K).add(eps + K * negative_denom_pre) # calculate denom part of neg probs
        probs_negative = probs_negative_denom.clone().fill_(K * negative_denom_pre)
        final_negative_probs = 1 - torch.div(probs_negative, probs_negative_denom) # neg pair probs (1-pn)
     
        final_positive_probs.log_() # take logs for positive sample probs
        final_negative_probs.log_() # take logs for negative sample probs
        
        sum_pos = final_positive_probs.sum(0)
        sum_neg = final_negative_probs.view(-1, 1).sum(0)
        
        loss = - (sum_pos + sum_neg) / batch_size
        
        return loss

