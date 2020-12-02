#!/usr/bin/env python3

import numpy as np, torch, sys, random

from game import Game, kW
from model import Model, ConvBlock, obs_to_torch
from config import Configs
from collections import Counter

device = torch.device('cuda')
kEnvs = 10000

if __name__ == "__main__":
    start_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2234
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    if sys.argv[1][-3:] == 'pkl': model.load_state_dict(torch.load(sys.argv[1])[0].state_dict())
    else: model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()
    envs = [Game(i + start_seed) for i in range(kEnvs)]
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
            _, _, over, info = envs[i].step((x[i], y[i]))
            if over:
                score[i] = info['score']
                finished[i] = True
    score = sorted(list(dict(Counter(score)).items()))
    for i, j in score: print(i, j)
    #score = [(i, j) for j, i in enumerate(score)]
