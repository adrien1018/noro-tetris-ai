#!/usr/bin/env python3

import pyautogui, time
import numpy as np

# Coordinates of the game field
# top-left
X1, Y1 = 207, 264           # change before use
# bottom-right
X2, Y2 = X1 + 180, Y1 + 360 # change before use

DX, DY = (X2 - X1) // 10, (Y2 - Y1) // 20

ITEM_THRESH = 150

def GetCoord(y, x):
    return int((X2 - X1) * (x + 0.5) / 10), int((Y2 - Y1) * (y + 0.5) / 20)

def GetKind(x): # 2*4 array
    if x[0][2] > 0: # ILOS
        if x[1][0] > 0: # LS
            if x[0][1] > 0: return 4 # S
            else          : return 5 # L
        else: # IO
            if x[0][0] > 0: return 6 # I
            else          : return 3 # O
    else: # IJTZ
        p = (x[0][0] > 0) * 2 + (x[0][1] > 0)
        if   p == 0: return 6 # I
        elif p == 1: return 0 # T
        elif p == 2: return 1 # J
        else       : return 2 # Z

def GetMap():
    x = np.array(pyautogui.screenshot(region = (X1, Y1, X2-X1, Y2-Y1))).max(axis = 2)
    return x[DY//2::DY, DX//2::DX]

def GetQueue():
    x = np.array(pyautogui.screenshot(region = (X2 + DX, Y1 + DY, DX*4, DY*14))).max(axis = 2)
    x = x[DY//2::DY, DX//2+DX//5::DX]
    return [GetKind(x[i*3:i*3+2]) for i in range(5)]

def Restart():
    pyautogui.press('f4')
    time.sleep(0.1)
    return GetQueue()

def WaitQueueChanged(prev_queue):
    start = time.time()
    while True:
        v = GetQueue()
        if v != prev_queue: return v
        if time.time() - start > 20: return

def Move(d):
    if d == 'R':
        pyautogui.platformModule._keyDown('right')
        pyautogui.platformModule._keyUp('right')
    elif d == 'L':
        pyautogui.platformModule._keyDown('left')
        pyautogui.platformModule._keyUp('left')
    elif d == 'H': pyautogui.press('c') # Hold
    elif d == 'F': pyautogui.press('space')
    elif d == 'D':
        pyautogui.platformModule._keyDown('down')
        time.sleep(0.005)
        pyautogui.platformModule._keyUp('down')
    elif d == 'DD':
        pyautogui.platformModule._keyDown('down')
        time.sleep(0.1)
        pyautogui.platformModule._keyUp('down')
    else:
        raise Exception()

import sys, torch, os, datetime
import tetris
import tetris.Tetris_Internal
from game import kH, kW, Game
from model import Model, obs_to_torch
from config import Configs

def Loop(model):
    game = Game(0)
    queue = Restart()
    game.reset(queue = queue)
    queue = WaitQueueChanged(queue)
    game.set_queue(queue)
    score = 0
    while True:
        with torch.no_grad():
            pi = model(obs_to_torch(game.obs).unsqueeze(0))[0]
            act = torch.argmax(pi.probs, 1).item()
        orig_board = np.array(game.env.board)
        board = None
        cur = game.env.cur
        allowed, t_dir = tetris.GetAllowed(orig_board, cur, False, 6, True)
        _, _, over, info = game.step(act)
        if act >= kH * kW:
            board = GetMap() > ITEM_THRESH
            if (board - orig_board[1:] < 0).any(): return
            time.sleep(0.1)
            Move('H')
            queue = [game.env.queue[i] for i in range(5)]
            continue
        else:
            x, y = act // kW, act % kW
            moves = tetris.Tetris_Internal.Moves(t_dir, cur, False, 0, x, y)
            now_x, now_y = 1, 4
            for px, move in moves:
                if px > now_x:
                    while True:
                        flag = board is None
                        board = GetMap() > ITEM_THRESH
                        if flag:
                            if (board - orig_board[1:] < 0).any(): 
                                return
                        diff = np.argwhere(board - orig_board[1:] > 0)
                        if diff.size == 0: return
                        now_x = diff[-1,0] + 1
                        if px <= now_x: break
                        if px - now_x > 8 or allowed[0,px,now_y] > 0:
                            Move('DD')
                        elif px - now_x > 1:
                            Move('D')
                Move('..LR..'[move])
                if move == 2: now_y -= 1
                elif move == 3: now_y += 1
            time.sleep(0.1)
            Move('F')
            print(datetime.datetime.now(), game.env.lines, flush = True)
        if over:
            print(datetime.datetime.now(), info, flush = True)
            return
        queue = WaitQueueChanged(queue)
        if not queue: return
        game.set_queue(queue)

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels, c.blocks).cuda()
    model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth') if len(sys.argv) <= 1 else sys.argv[1]
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()
    a = model(obs_to_torch(np.stack([Game(2).obs])))[1].sum().item()
    print('Start')
    while True:
        try: Loop(model)
        except KeyboardInterrupt: break
        except: pass
        time.sleep(4)

