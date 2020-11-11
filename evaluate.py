#!/usr/bin/env python3

import numpy as np, torch, sys, random
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from labml.configs import BaseConfigs

import tetris

device = torch.device('cuda')
# board, valid, current(7), next(7)
kInChannel = 1 + 1 + 7 + 7
kH, kW = 20, 10
kTensorDim = (3, kH, kW)
kMaxFail = 3

class Game:
    def __init__(self, seed: int):
        self.args = (0, False)
        self.env = tetris.Tetris(*self.args)
        self.env.Seed(seed)
        # board, current(7), next(7), speed(14)
        self.obs = np.zeros(kTensorDim, dtype = np.uint8)
        self.set_obs()
        # keep track of the episode rewards
        self.rewards = []
        self.cnt = 0

    def set_obs(self):
        self.obs[:] = 0
        self.obs[0] = 1 - self.env.board
        self.obs[1] = self.env.allowed[0]
        self.obs[2,0,0] = self.env.cur
        self.obs[2,0,1] = self.env.nxt

    def step(self, action):
        """
        Executes `action` (x, y)
         returns a tuple of (observation, reward, done, info).
        """
        suc, score, _ = self.env.Place(*action)
        reward = score if suc else -0.1
        self.cnt = 0 if suc else self.cnt + 1
        self.set_obs()
        # maintain rewards for each step
        self.rewards.append(reward)
        over = self.env.over or self.cnt >= kMaxFail

        if over:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None
        return self.obs, reward, over, episode_info

    def reset(self):
        """ Reset environment """
        self.env.Reset(*self.args)
        self.set_obs()
        self.rewards = []
        self.cnt = 0
        return self.obs

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
        # A fully connected layer to get logits for $\pi$
        self.pi_logits_head = nn.Sequential(
                nn.Conv2d(ch, 4, 1),
                nn.BatchNorm2d(4),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(4 * kH * kW, kH * kW)
                )
        # A fully connected layer to get value function
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
        q = torch.zeros((obs.shape[0], kInChannel, kH, kW), dtype = torch.float32, device = device)
        q[:,0:2] = obs[:,0:2]
        q.scatter_(1, (2 + obs[:,2,0,0].type(torch.long)).view(-1, 1, 1, 1).repeat(1, 1, kH, kW), 1)
        q.scatter_(1, (9 + obs[:,2,0,1].type(torch.long)).view(-1, 1, 1, 1).repeat(1, 1, kH, kW), 1)
        x = self.start(q)
        x = self.res(x)
        pi = self.pi_logits_head(x)
        pi -= (1 - obs[:,1].view(-1, kH * kW)) * 20
        value = self.value(x).reshape(-1)
        pi_sample = Categorical(logits = torch.clamp(pi, -30, 30))
        return pi_sample, value

def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.uint8, device = device)

class Configs(BaseConfigs):
    # #### Configurations
    # $\gamma$ and $\lambda$ for advantage calculation
    gamma: float = 0.996
    lamda: float = 0.95
    # number of updates
    updates: int = 80000
    # number of epochs to train the model with sampled data
    epochs: int = 2
    # number of worker processes
    n_workers: int = 2
    env_per_worker: int = 16
    # number of steps to run on each process for a single update
    worker_steps: int = 128
    # size of mini batches
    mini_batch_size: int = 512
    channels: int = 128
    blocks: int = 8
    start_lr: float = 2e-4
    start_clipping_range: float = 0.2
    vf_weight: float = 0.5
    entropy_weight: float = 1e-2

kEnvs = 500

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels.value, c.blocks.value).to(device)
    model.load_state_dict(torch.load(sys.argv[1])[0].state_dict())
    model.eval()
    envs = [Game(random.randint(0, 65535)) for i in range(kEnvs)]
    finished = [False for i in range(kEnvs)]
    score = [0. for i in range(kEnvs)]
    while not all(finished):
        obs = []
        for i in envs: obs.append(i.obs)
        with torch.no_grad():
            obs = obs_to_torch(np.stack(obs))
            pi = model(obs)[0]
            act = torch.argmax(pi.probs, 1).cpu().numpy()
            #act = pi.sample().cpu().numpy()
        x, y = act // kW, act % kW
        tb = []
        for i in range(kEnvs):
            if finished[i]: continue
            _, reward, over, _ = envs[i].step((x[i], y[i]))
            score[i] += reward
            if over: finished[i] = True
    score = [(i, j) for j, i in enumerate(score)]
    score.sort()
    print('mn: %.1f' % score[0][0])
    for i in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        print('%.2f: %.1f' % (i, score[int(i * kEnvs)][0]))
    print('mx: %.1f' % score[-1][0])
