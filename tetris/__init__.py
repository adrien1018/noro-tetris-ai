from .Tetris_Internal import *
import numpy as np
import random

kCumW = [33, 65, 97, 129, 162, 193, 224]
kLimits = [9, 8, 7, 6, 5, 4, 3, 3, 2, 2, 2, 2, 2]
kSpeed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12,
          13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
kScore = [0, 40, 100, 300, 1200]

def GetAllowed(board, k, rot, lim):
    ret = np.zeros((4 if rot else 1, 20, 10), dtype = 'int32')
    Allowed(board.data, k, rot, lim, ret.data)
    return ret

def GetRand(k = 1):
    return random.choices(range(7), cum_weights = kCumW, k = k)

class Tetris:
    def __init__(self, start = 0, rotate = True):
        self.Reset(start, rotate)

    def Reset(self, start = 0, rotate = True):
        self.board = np.zeros((20, 10), dtype = 'int32')
        self.cur, self.nxt = GetRand(2)
        self.lines = 0
        self.start = start
        self.rotate = rotate
        self._SetInternal()
        self.over = False
        self.random = random.Random()

    def Seed(self, seed):
        self.random.seed(seed)

    def Speed(self):
        return kSpeed[self.level] if self.level <= 28 else 14

    def _SetInternal(self):
        # level
        t_offset = (min(10, (self.start + 1) % 16) + (self.start + 1) // 16 * 10) * 10
        self.level = self.start if self.lines < t_offset else self.start + (self.lines - t_offset) // 10 + 1
        # allowed
        lim = 1 if self.level >= 13 else kLimits[self.level]
        self.allowed = GetAllowed(self.board, self.cur, self.rotate, lim)
        self.over = not np.any(self.allowed)

    def Place(self, a1, a2, a3 = None):
        if self.over: return False, None, None
        if self.rotate: g, x, y = a1, a2, a3
        else: g, x, y = 0, a1, a2
        if self.allowed[g,x,y] == 0: return False, None, None
        num = Place(self.board.data, self.cur, g, x, y, 1, True)
        self.lines += num
        self.cur, self.nxt = self.nxt, GetRand()[0]
        self._SetInternal()
        return True, num, (self.level + 1) * kScore[num]
