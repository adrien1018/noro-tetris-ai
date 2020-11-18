import multiprocessing, hashlib
import numpy as np

import tetris

kH, kW = 20, 10
kTensorDim = (3, kH, kW)
kMaxFail = 3

def CalReward(success, score, new_score):
    if not success: return -0.125
    #return (score + new_score) ** 2 - score ** 2
    ret = 0
    for i in range(new_score):
        if score + i < 30: ret += 0.5
        elif score + i < 35: ret += 3
        else: ret += 8
    return ret

class Game:
    def __init__(self, seed: int, tpow = 1):
        self.args = (0, False)
        self.tpow = tpow
        self.obs = np.zeros(kTensorDim, dtype = np.uint8)
        self.env = tetris.Tetris(seed, *self.args)
        self.reset(False)

    def set_state(self, cur, nxt, board = None):
        if board is None: board = np.zeros((20, 10), dtype = 'int32')
        self.env.board = board.astype('int32')
        self.env.cur = cur
        self.env.nxt = nxt
        self.env._SetInternal()
        self._set_obs()

    def _set_obs(self):
        self.obs[:] = 0
        self.obs[0] = 1 - self.env.board
        self.obs[1] = self.env.allowed[0]
        self.obs[2,0,0] = self.env.cur
        self.obs[2,0,1] = self.env.nxt
        self.obs[2,0,2] = self.score

    def step(self, action, **kwargs):
        """
        Executes `action` (x, y)
         returns a tuple of (observation, reward, done, info).
        """
        suc, score, _ = self.env.Place(*action, **kwargs)
        reward = CalReward(suc, self.score, score)
        if suc: self.score += score
        self.reward += reward
        self.length += 1
        self.fail_cnt = 0 if suc else self.fail_cnt + 1
        self._set_obs()
        over = self.env.over or self.fail_cnt >= kMaxFail

        if over:
            # if finished, set episode information if episode is over, and reset
            episode_info = {'score': self.score, 'reward': self.reward, 'length': self.length}
            self.reset()
        else:
            episode_info = None
        return self.obs, reward, over, episode_info

    def reset(self, env_reset = True):
        """ Reset environment """
        if env_reset: self.env.Reset(*self.args)
        self.fail_cnt = 0
        self.length = 0
        self.reward = 0
        self.score = 0
        self._set_obs()
        return self.obs

def worker_process(remote: multiprocessing.connection.Connection, seed: int, num: int):
    """Each worker process runs this method"""
    # create game
    Seed = lambda x: int.from_bytes(hashlib.sha256(
        int.to_bytes(seed, 4, 'little') + int.to_bytes(x, 4, 'little')).digest(), 'little')
    games = [Game(Seed(i)) for i in range(num)]
    # wait for instructions from the connection and execute them
    while True:
        result = []
        cmd, data = remote.recv()
        if cmd == "step":
            for i in range(num): result.append(games[i].step((data[i] // kW, data[i] % kW)))
            obs, rew, over, info = zip(*result)
            remote.send((np.stack(obs), np.stack(rew), np.array(over), list(info)))
        elif cmd == "reset":
            for i in range(num): result.append(games[i].reset())
            remote.send(np.stack(result))
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker:
    """Creates a new worker and runs it in a separate process."""
    def __init__(self, seed, num):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target = worker_process, args = (parent, seed, num))
        self.process.start()
