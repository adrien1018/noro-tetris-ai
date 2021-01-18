#!/usr/bin/env python3

import numpy as np, torch, sys, random, time, os.path
import tetris

from game import Game, kH, kW, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetTorch(game):
    return obs_to_torch(game.obs).unsqueeze(0)

def Print(board, *args):
    print(*args)
    input()
    if isinstance(board, str):
        print(board)
    else:
        print('\n'.join([' '.join([str(board[i,j]) for j in range(kW)]) for i in range(1,kH)]))

def GetStrat(game):
    with torch.no_grad():
        pi = model(GetTorch(game))[0]
        act = torch.argmax(pi.probs, 1).item()
        return act

kStr = [['  #    ', '#      ', '# #    ', '  # #  ', '  # #  ', '    #  ', '# # # #'],
        ['# # #  ', '# # #  ', '  # #  ', '  # #  ', '# #    ', '# # #  ', '       ']]
def GetLine(game, l):
    return '|'.join([kStr[l][i] for i in [game.env.cur] + list(game.env.queue)[:5]]) \
            + ('||' + kStr[l][game.env.hold] if game.env.hold is not None else '')
def PrintStrat(game):
    act = GetStrat(game)
    if act >= kW * kH:
        board = 'Hold'
    else:
        board = np.array(game.env.board)
        tetris.Tetris_Internal.Place(board.data, game.env.cur, 0, act // kW, act % kW, 2)
    Print(board, GetLine(game, 0) + '\n' + GetLine(game, 1), game.score)
    print(act // kW, act % kW)
    return act

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth') if len(sys.argv) <= 1 else sys.argv[1]
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
            act = PrintStrat(game)
            if game.step(act)[2]: break
            input()
