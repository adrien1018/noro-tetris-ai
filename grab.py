from PIL import Image, ImageGrab
from ctypes import windll
import win32gui, win32ui, xxhash
import numpy as np
import time

class Capture:
    def __init__(self):
        toplist, winlist = [], []
        def enum_cb(hwnd, results):
            winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
        win32gui.EnumWindows(enum_cb, toplist)

        self.hwnd = [(hwnd, title) for hwnd, title in winlist if 'fceux' in title.lower()][0][0]

    def Capture(self):
        while True:
            try:
                left, top, right, bot = win32gui.GetClientRect(self.hwnd)
                w = right - left
                h = bot - top

                hwndDC = win32gui.GetWindowDC(self.hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()

                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

                saveDC.SelectObject(saveBitMap)

                result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 1)

                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)

                dimen = (bmpinfo['bmWidth'], bmpinfo['bmHeight'])
                if dimen != (512, 448): continue
                self.im = Image.frombuffer('RGB', dimen,
                    bmpstr, 'raw', 'BGRX', 0, 1)
                break
            except KeyboardInterrupt: raise
            except: pass

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        self.arr = np.array(self.im)
        return self.arr

    def GetBoard(self):
        kX, kY = 85, 197
        return (self.arr[kX:kX+320:16, kY:kY+160:16].sum(axis = 2) > 0).astype('int32')

    @staticmethod
    def GetType(sub):
        s = sub.sum(axis = 1)
        if s[1] == 0: return 6 # I
        if s[1] == 1:
            if sub[1][0]: return 5 # L
            if sub[1][1]: return 0 # T
            return 1 # J
        if sub[0][0] == sub[1][0]: return 3 # O
        if sub[0][0]: return 2 # Z
        return 4 # S

    def GetNext(self):
        kX, kY = 234, 393
        sub = (self.arr[kX:kX+32:16, kY:kY+48:16].sum(axis = 2) > 0).astype('int32')
        return Capture.GetType(sub)

    def GetMode(self):
        return xxhash.xxh64(np.array(self.arr[40:72,40:152]).data).hexdigest()

    kBlocks = np.array([
        [[0, 1, 1, 1], [0, 0, 1, 0]],
        [[0, 1, 1, 1], [0, 0, 0, 1]],
        [[0, 1, 1, 0], [0, 0, 1, 1]],
        [[0, 1, 1, 0], [0, 1, 1, 0]],
        [[0, 0, 1, 1], [0, 1, 1, 0]],
        [[0, 1, 1, 1], [0, 1, 0, 0]],
        [[1, 1, 1, 1], [0, 0, 0, 0]]
    ])

import sys, torch
from model import Model, ConvBlock, obs_to_torch
from config import Configs
from game import kH, kW
import tetris
import tetris.Tetris_Internal

def GetMovesString(m):
    ret = '---' if len(m) == 0 or m[0][0] != 0 else ''
    lst = []
    for i, j in m:
        if len(lst) == 0 or lst[-1][0] != i: lst.append((i, []))
        lst[-1][1].append('.DLRAB'[j])
    ret += ' '.join(['{}:'.format(i) + ''.join(j) for i, j in lst])
    return ret

kTransition = [
    [1, 5, 6, 5, 5, 5, 5],
    [6, 1, 5, 5, 5, 5, 5],
    [5, 6, 1, 5, 5, 5, 5],
    [5, 5, 5, 2, 5, 5, 5],
    [5, 5, 5, 5, 2, 5, 5],
    [6, 5, 5, 5, 5, 1, 5],
    [5, 5, 5, 5, 6, 5, 1],
]

def PrintStrat(model, board, now, nxt, score, tx = None, ty = None):
    obs = np.zeros((3, kH, kW), dtype = 'uint8')
    obs[0] = 1 - board
    obs[1], t_dir = tetris.GetAllowed(board, now, False, 9, True)
    obs[2,0,0] = now
    obs[2,0,1] = nxt
    obs[2,0,2] = score
    with torch.no_grad():
        pi = model(obs_to_torch(obs).unsqueeze(0))[0]
        act = torch.argmax(pi.probs, 1).item()
        x, y = act // kW, act % kW
    s = board.astype('int32')
    tetris.Tetris_Internal.Place(s.data, now, 0, x, y, 2, False)
    print(s)
    print(GetMovesString(tetris.Tetris_Internal.Moves(t_dir, now, False, 0, x, y)))
    sys.stdout.flush()

    fx, fy = None, None
    pos = [(x, y)]
    if tx is not None: pos.append((tx, ty))
    cols = []
    for rx, ry in pos:
        s = board.astype('int32')
        dscore = score + tetris.Tetris_Internal.Place(s.data, now, 0, rx, ry, 1, True)
        obs = np.zeros((3, kH, kW), dtype = 'uint8')
        obs[0] = 1 - s
        obs[1], t_dir = tetris.GetAllowed(s, nxt, False, 9, True)
        obs[2,0,0] = nxt
        obs[2,0,2] = dscore
        obs = np.tile(obs, (7, 1, 1, 1))
        for i in range(7): obs[i,2,0,1] = i
        with torch.no_grad():
            pi = model(obs_to_torch(obs))[0]
            act = torch.argmax(pi.probs, 1).cpu().numpy()
            gx, gy = act // kW, act % kW
        mp = {}
        for i in range(7):
            r = (gx[i], gy[i])
            if r not in mp: mp[r] = 0
            mp[r] += kTransition[nxt][i]
        mp = sorted(mp.items(), key = lambda x: -x[1])
        arr = []
        for r, prob in mp:
            moves = GetMovesString(tetris.Tetris_Internal.Moves(t_dir, nxt, False, 0, r[0], r[1]))
            arr.append('[next] %2d/32 -> %s' % (prob, moves))
        cols.append(arr)
        if fx is None:
            fx, fy = mp[0][0]
            print(fx, fy)
    if len(cols) == 1: cols.append([])
    while len(cols[0]) < len(cols[1]): cols[0].append('')
    while len(cols[1]) < len(cols[0]): cols[1].append('')
    for i in zip(*cols): print('%-35s%s' % i)
    sys.stdout.flush()
    return fx, fy

def Loop(model):
    print('Ready', flush = True)
    cap = Capture()
    cap.Capture()
    if cap.GetMode() == 'dc25f53685046447': return
    while True:
        cap.Capture()
        if cap.GetMode() == 'dc25f53685046447': break
    while True:
        board = cap.GetBoard()
        if board.sum() > 0: break
        cap.Capture()
    now = Capture.GetType(board[0:2,4:7])
    nxt = cap.GetNext()
    assert (board[0:2,3:7] - Capture.kBlocks[now]).min() == 0
    board[0:2,3:7] -= Capture.kBlocks[now]
    x, y = PrintStrat(model, board, now, nxt, 0)
    prev = cap.arr[176:384,128:144]
    N = 0
    while True:
        while True:
            arr = cap.Capture()
            if not (arr[176:384,128:144] == prev).all() and \
                arr.sum() > 0: break
        if cap.GetMode() != 'dc25f53685046447': return
        N += 1
        now, nxt = nxt, cap.GetNext()
        board = cap.GetBoard()
        assert (board[0:2,3:7] - Capture.kBlocks[now]).min() == 0
        board[0:2,3:7] -= Capture.kBlocks[now]
        score = (N * 4 - board.sum()) // 10
        x, y = PrintStrat(model, board, now, nxt, score, x, y)
        prev = arr[176:384,128:144]

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels, c.blocks).cuda()
    if sys.argv[1][-3:] == 'pkl': model.load_state_dict(torch.load(sys.argv[1])[0].state_dict())
    else: model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()
    while True:
        Loop(model)

