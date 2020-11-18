#!/usr/bin/env python3

import numpy as np, torch, sys, random, time
import tetris

from game import Game, kW, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda')

def GetTorch(game):
    return obs_to_torch(game.obs).unsqueeze(0)

def Print(board, *args):
    print(*args)
    input()
    print('\n'.join([' '.join([str(board[i,j]) for j in range(10)]) for i in range(20)]))

def GetStrat(game):
    with torch.no_grad():
        pi = model(GetTorch(game))[0]
        act = torch.argmax(pi.probs, 1).item()
        return act // kW, act % kW

kStr = [['# # #  ', '# # #  ', '# #    ', '  # #  ', '  # #  ', '# # #  ', '# # # #'],
        ['  #    ', '    #  ', '  # #  ', '  # #  ', '# #    ', '#      ', '       ']]
def PrintStrat(game):
    x, y = GetStrat(game)
    board = np.array(game.env.board)
    tetris.Tetris_Internal.Place(board.data, game.env.cur, 0, x, y, 2)
    t0 = kStr[0][game.env.cur] + '|' + kStr[0][game.env.nxt]
    t1 = kStr[1][game.env.cur] + '|' + kStr[1][game.env.nxt]
    Print(board, t0 + '\n' + t1)
    return x, y

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    if sys.argv[1][-3:] == 'pkl': model.load_state_dict(torch.load(sys.argv[1])[0].state_dict())
    else: model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()
    for i in range(1000):
        game = Game(random.randint(0, 10000000))
        while True:
            _, _, x, y = game.step(GetStrat(game))
            if x: break
        if y['score'] < 25: continue
        game = Game(i)
        while True:
            x, y = PrintStrat(game)
            if game.step((x, y))[2]: break
            input()
