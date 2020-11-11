#!/usr/bin/env python3

import numpy as np, torch, sys, random
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from labml.configs import BaseConfigs

from game import Game, kW
from model import Model, ConvBlock

device = torch.device('cuda')

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
    envs = [Game(random.randint(0, 2**32-1)) for i in range(kEnvs)]
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
