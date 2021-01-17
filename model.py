import torch, numpy as np
from torch import nn
from torch.distributions import Categorical
from torch.cuda.amp import autocast

from game import kH, kW

# board, valid, current(7), next(7), score
#kInChannel = 1 + 1 + 7 + 7 + 1

# board(1),valid(1),current(7),next(7*5),step%7(7),hold(8),holded(1),score(1)
kInChannel = 1 + 1 + 7 + 7 * 5 + 7 + 8 + 1 + 1

class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                )
        self.final = nn.ReLU(True)
    def forward(self, x):
        return self.final(self.main(x) + x)

class Model(nn.Module):
    def __init__(self, ch, blk):
        super().__init__()
        self.start = nn.Sequential(
                nn.Conv2d(kInChannel, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                )
        self.res = nn.Sequential(*[ConvBlock(ch) for i in range(blk)])
        self.pi_logits_head = nn.Sequential(
                nn.Conv2d(ch, 4, 1),
                nn.BatchNorm2d(4),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(4 * kH * kW, kH * kW + 1 + 3) # +1 for hold, +3 for padding
                )
        self.value = nn.Sequential(
                nn.Conv2d(ch, 1, 1),
                nn.BatchNorm2d(1),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(1 * kH * kW, 256),
                nn.ReLU(True),
                nn.Linear(256, 1)
                )

    @autocast()
    def forward(self, obs: torch.Tensor):
        q = torch.zeros((obs.shape[0], kInChannel, kH, kW), dtype = torch.float32, device = obs.device)
        q[:,0:2] = obs[:,0:2]
        for i in range(8): # the last one is 8
            q.scatter_(1, (2 + i*7 + obs[:,2,0,i].type(torch.long)).view(-1, 1, 1, 1).repeat(1, 1, kH, kW), 1)
        q[:,59] = obs[:,2,0,8].view(-1, 1, 1)
        q[:,60] = (obs[:,2,0,9] / 32).view(-1, 1, 1)
        x = self.start(q)
        x = self.res(x)
        pi = self.pi_logits_head(x)

        mp = torch.zeros((obs.shape[0], kH * kW + 4), dtype = torch.float32, device = obs.device)
        mp[:,:kH*kW] = obs[:,1].view(-1, kH * kW)
        mp[:,kH*kW] = 1 - obs[:,2,0,8] # 1: holded, cannot hold again
        if self.training:
            pi -= (1 - mp) * 20
        else:
            pi[mp == 0] = -30
        value = self.value(x).reshape(-1)
        pi_sample = Categorical(logits = torch.clamp(pi, -30, 30))
        return pi_sample, value

def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.uint8, device = torch.device('cuda'))
