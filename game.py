import multiprocessing, hashlib
import numpy as np

import tetris

kH, kW = 20, 10
kTensorDim = (3, kH, kW)
kMaxFail = 3

class Game:
    def __init__(self, seed: int):
        self.args = (0, False)
        self.env = tetris.Tetris(*self.args)
        self.env.Seed(seed)
        # board, current(7), next(7), speed(14)
        self.obs = np.zeros(kTensorDim, dtype = np.uint8)
        self.set_obs()
        # keep track of the episode rewards
        self.rewards = []
        self.cnt = 0

    def set_obs(self):
        self.obs[:] = 0
        self.obs[0] = 1 - self.env.board
        self.obs[1] = self.env.allowed[0]
        self.obs[2,0,0] = self.env.cur
        self.obs[2,0,1] = self.env.nxt

    def step(self, action):
        """
        Executes `action` (x, y)
         returns a tuple of (observation, reward, done, info).
        """
        suc, score, _ = self.env.Place(*action)
        reward = score if suc else -0.1
        self.cnt = 0 if suc else self.cnt + 1
        self.set_obs()
        # maintain rewards for each step
        self.rewards.append(reward)
        over = self.env.over or self.cnt >= kMaxFail

        if over:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None
        return self.obs, reward, over, episode_info

    def reset(self):
        """ Reset environment """
        self.env.Reset(*self.args)
        self.set_obs()
        self.rewards = []
        self.cnt = 0
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
