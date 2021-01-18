from .Tetris_Internal import *
import numpy as np
import random
from collections import deque

kLimits = [9, 8, 7, 6, 5, 4, 3, 3, 2, 2, 2, 2, 2]
kSpeed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12,
          13, 13, 13, 13, 13, 13, 13, 13, 13, 13]
kScore = [0, 40, 100, 300, 1200]
kH, kW = 21, 10

def GetAllowed(board, k, rot, lim, with_dir = False):
    ret = np.zeros((4 if rot else 1, kH, kW), dtype = 'int32')
    if with_dir:
        ret_dir = np.zeros_like(ret)
        Allowed(board.data, k, rot, lim, ret.data, ret_dir.data)
        return ret, ret_dir
    else:
        Allowed(board.data, k, rot, lim, ret.data)
        return ret

def GetRand(rand, prev = None):
    if prev is None: prev = rand.randint(0, 6)
    res = rand.randint(0, 7)
    if res != 7 and res != prev: return res
    kTable = [2, 0, 1, 3, 4, 0, 4]
    return (rand.randint(0, 7) + kTable[prev]) % 7

def GetRandPerm(rand):
    ret = list(range(7))
    rand.shuffle(ret)
    return ret

class Tetris:
    def __init__(self, seed = None, start = 0, rotate = True, cur = None, nxt = None):
        self.random = random.Random()
        if seed is not None: self.Seed(seed)
        self.Reset(start, rotate)

    def _QueueStep(self):
        if len(self.queue) < 7: self.queue.extend(GetRandPerm(self.random))
        return self.queue.popleft()

    def Reset(self, start = 0, rotate = True, cur = None, nxt = None):
        self.board = np.zeros((kH, kW), dtype = 'int32')
        self.steps = 0
        self.queue = deque() #
        self.cur = self._QueueStep() #
        self.hold = None
        self.holded = False
        #self.cur = GetRand(self.random) if cur is None else cur
        #self.nxt = GetRand(self.random, self.cur) if nxt is None else nxt
        self.lines = 0
        self.start = start
        self.rotate = rotate
        self._SetInternal()
        self.over = False

    def Hold(self):
        if self.holded: return False, None, None
        if self.hold is None:
            self.hold = self.cur
            self.cur = self._QueueStep()
        else:
            self.hold, self.cur = self.cur, self.hold
        self.holded = True
        return True, 0, 0

    def Seed(self, seed):
        self.random.seed(seed)

    def Speed(self):
        return kSpeed[self.level] if self.level <= 28 else 14

    def _SetInternal(self):
        # level
        t_offset = (min(10, (self.start + 1) % 16) + (self.start + 1) // 16 * 10) * 10
        self.level = self.start if self.lines < t_offset else self.start + (self.lines - t_offset) // 10 + 1
        # allowed
        #lim = 1 if self.level >= 13 else kLimits[self.level]
        self.allowed = GetAllowed(self.board, self.cur, self.rotate, 6)
        self.over = not np.any(self.allowed)

    def Place(self, a1, a2, a3 = None, t_nxt = None):
        if self.over: return False, None, None
        if self.rotate: g, x, y = a1, a2, a3
        else: g, x, y = 0, a1, a2
        if self.allowed[g,x,y] == 0: return False, None, None
        num = Place(self.board.data, self.cur, g, x, y, 1, True)
        self.lines += num
        self.steps += 1
        self.holded = False
        self.cur = self._QueueStep() #
        #self.cur, self.nxt = self.nxt, (GetRand(self.random, self.nxt) if t_nxt is None else t_nxt)
        self._SetInternal()
        return True, num, (self.level + 1) * kScore[num]
