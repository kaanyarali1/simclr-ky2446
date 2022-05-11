import torch
import torch.nn as nn


"""
this was an initial attempt to implement NX-Tent loss function. This is not used.
this can be ignored!
"""
def nt_xent_loss(output1, output2, temperature):

	out =  torch.cat([output1,output2],dim = 0)
	n_samples = len(out)
	similarity_func = nn.CosineSimilarity(dim=2)

	cov = similarity_func(out.unsqueeze(1), out.unsqueeze(0))
	sim =  torch.exp(cov/temperature)

	mask = torch.eye(n_samples, device = sim.device).bool()
	neg = sim.masked_select(mask).view(n_samples,-1).sum(dim=-1)

	pos_sim = nn.CosineSimilarity()
	pos = torch.exp(pos_sim(output1,output2) / temperature)
	pos = torch.cat([pos,pos], dim = 0)

	loss = -torch.log(pos/neg).mean()

	return loss




