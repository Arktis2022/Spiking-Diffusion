import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PSP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau_s = 2

    def forward(self, inputs):
        """
        inputs: (T, N, C)
        """
        syns = None
        syn = 0
        n_steps = inputs.shape[0]
        
        for t in range(n_steps):
            syn = syn + (inputs[t,...] - syn) / self.tau_s
            if syns is None:
                syns = syn.unsqueeze(0)
            else:
                syns = torch.cat([syns, syn.unsqueeze(0)], dim=0)
            #print (syn.shape)
        return syns

class MembraneOutputLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_steps = 16

        arr = torch.arange(n_steps-1,-1,-1)
        self.register_buffer("coef", torch.pow(0.8, arr)[:,None,None,None,None]) # (T,1,1,1,1)

    def forward(self, x):
        """
        x : (T,N,C,H,W)
        """
        out = torch.sum(x*self.coef, dim=0)
        return out
