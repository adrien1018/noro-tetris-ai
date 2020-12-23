#!/usr/bin/env python3

import numpy as np, torch, sys, random, time
import tetris

from game import Game, kW, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    Print(board, t0 + '\n' + t1, game.score)
    return x, y

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    model_path = 'models/model.pth' if len(sys.argv) <= 1 else sys.argv[1]
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()
    while True:
        seed = random.randint(0, 2**32-1)
        game = Game(seed)
        while True:
            _, _, x, y = game.step(GetStrat(game))
            if x: break
        if y['score'] < 1: continue
        game = Game(seed)
        while True:
            x, y = PrintStrat(game)
            if game.step((x, y))[2]: break
            input()
