from torch import nn
import torch
import numpy as np

class DynamicSimCLR_Loss(nn.Module):
    
    """same sa SIMCLR Loss but without the need of batchsize"""
    
    def __init__(self, temperature):
        super().__init__()
        #self.batch_size = batch_size
        self.temperature = temperature
        self.mask=None
        #
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        
        batch_size=z_i.shape[0]
        self.batch_size=batch_size
        
        self.mask = self.mask_correlated_samples(batch_size)
        
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)
        
        
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        #print(sim_i_j.shape,sim_j_i.shape)
        
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        
        #print(f"[DEBUG] pos: {positive_samples.shape} neg: {negative_samples.shape} labels: {labels.shape} logits: {logits.shape}")
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss