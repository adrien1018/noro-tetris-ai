#!/usr/bin/env python3

import sys, traceback, os, subprocess, collections
from typing import Dict, List
from sortedcontainers import SortedList
import numpy as np, torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from labml import monit, tracker, logger, experiment
from labml.configs import BaseConfigs

from game import Game, Worker, kTensorDim
from model import Model, ConvBlock

device = torch.device('cuda')

def OutputSMI(msg):
    x = subprocess.run("nvidia-smi | grep python3", shell = True, capture_output = True)
    print(msg, os.getpid(), '\n' + x.stdout.decode(), flush = True)

def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.uint8, device = device)

class Configs(BaseConfigs):
    # #### Configurations
    # $\gamma$ and $\lambda$ for advantage calculation
    gamma: float = 0.996
    lamda: float = 0.95
    # number of updates
    updates: int = 80000
    # number of epochs to train the model with sampled data
    epochs: int = 2
    # number of worker processes
    n_workers: int = 2
    env_per_worker: int = 16
    # number of steps to run on each process for a single update
    worker_steps: int = 128
    # size of mini batches
    mini_batch_size: int = 512
    channels: int = 128
    blocks: int = 8
    start_lr: float = 2e-4
    start_clipping_range: float = 0.2
    vf_weight: float = 0.5
    entropy_weight: float = 1e-2

class SortedQueue:
    def __init__(self, sz):
        self.size = sz
        self.__list = SortedList()
        self.__queue = collections.deque()
    def add(self, val):
        self.__list.add(val)
        self.__queue.append(val)
        if len(self.__queue) > self.size: self.__list.discard(self.__queue.popleft())
    def __len__(self): return len(self.__queue)
    def get_ratio(self, val):
        ind = min(len(self.__queue) - 1, max(0, int(len(self.__queue) * val)))
        return self.__list[ind]

class Main:
    def __init__(self, c: Configs):
        self.c = c
        # total number of samples for a single update
        self.envs = self.c.n_workers * self.c.env_per_worker
        self.batch_size = self.envs * self.c.worker_steps
        assert (self.batch_size % self.c.mini_batch_size == 0)

        # #### Initialize
        # create workers
        self.workers = [Worker(47 + i, c.env_per_worker) for i in range(self.c.n_workers)]
        self.reward_queue = SortedQueue(400)

        # initialize tensors for observations
        self.obs = np.zeros((self.envs, *kTensorDim))
        for worker in self.workers:
            worker.child.send(("reset", None))
        for w, worker in enumerate(self.workers):
            self.obs[self.w_range(w)] = worker.child.recv()
        self.obs = obs_to_torch(self.obs)

        # model for sampling
        self.model = Model(c.channels, c.blocks).to(device)

        # optimizer
        self.scaler = GradScaler()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.c.start_lr)

    def w_range(self, x): return slice(x * self.c.env_per_worker, (x + 1) * self.c.env_per_worker)

    def sample(self) -> (Dict[str, np.ndarray], List):
        """### Sample data with current policy"""

        rewards = np.zeros((self.envs, self.c.worker_steps), dtype = np.float16)
        done = np.zeros((self.envs, self.c.worker_steps), dtype = np.bool)
        actions = torch.zeros((self.envs, self.c.worker_steps), dtype = torch.int32, device = device)
        obs = torch.zeros((self.envs, self.c.worker_steps, *kTensorDim), dtype = torch.uint8, device = device)
        log_pis = torch.zeros((self.envs, self.c.worker_steps), dtype = torch.float16, device = device)
        values = torch.zeros((self.envs, self.c.worker_steps), dtype = torch.float16, device = device)

        # sample `worker_steps` from each worker
        for t in range(self.c.worker_steps):
            with torch.no_grad():
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `n_workers`
                pi, v = self.model(self.obs)
                values[:, t] = v
                a = pi.sample()
                actions[:, t] = a
                log_pis[:, t] = pi.log_prob(a)

            # run sampled actions on each worker
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[self.w_range(w),t].cpu().numpy()))

            self.obs = np.zeros((self.envs, *kTensorDim))
            for w, worker in enumerate(self.workers):
                # get results after executing the actions
                now = self.w_range(w)
                self.obs[now], rewards[now,t], done[now,t], info_arr = worker.child.recv()

                # collect episode info, which is available if an episode finished;
                #  this includes total reward and length of the episode -
                #  look at `Game` to see how it works.
                # We also add a game frame to it for monitoring.
                for info in info_arr:
                    if not info: continue
                    self.reward_queue.add(info['reward'])
                    tracker.add('reward', info['reward'])
                    tracker.add('reward_per01', self.reward_queue.get_ratio(0.01))
                    tracker.add('reward_per10', self.reward_queue.get_ratio(0.1))
                    tracker.add('reward_per50', self.reward_queue.get_ratio(0.5))
                    tracker.add('reward_per90', self.reward_queue.get_ratio(0.9))
                    tracker.add('reward_per99', self.reward_queue.get_ratio(0.99))
                    tracker.add('length', info['length'])
            self.obs = obs_to_torch(self.obs)

        # calculate advantages
        advantages = self._calc_advantages(done, rewards, values)
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'log_pis': log_pis,
            'advantages': advantages
        }
        # samples are currently in [workers, time] table,
        #  we should flatten it
        for i in samples:
            samples[i] = samples[i].view(-1, *samples[i].shape[2:])
        return samples

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: torch.Tensor) -> torch.Tensor:
        """### Calculate advantages"""
        with torch.no_grad(), autocast():
            rewards = torch.from_numpy(rewards).to(device)
            done = torch.from_numpy(done).to(device)

            # advantages table
            advantages = torch.zeros((self.envs, self.c.worker_steps), dtype = torch.float16, device = device)
            last_advantage = torch.zeros(self.envs, dtype = torch.float16, device = device)

            # $V(s_{t+1})$
            _, last_value = self.model(self.obs)

            for t in reversed(range(self.c.worker_steps)):
                # mask if episode completed after step $t$
                mask = ~done[:, t]
                last_value = last_value * mask
                last_advantage = last_advantage * mask
                # $\delta_t$
                delta = rewards[:, t] + self.c.gamma * last_value - values[:, t]
                # $\hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}}$
                last_advantage = delta + self.c.gamma * self.c.lamda * last_advantage
                # note that we are collecting in reverse order.
                advantages[:, t] = last_advantage
                last_value = values[:, t]
        return advantages

    def train(self, samples: Dict[str, torch.Tensor], learning_rate: float, clip_range: float):
        """### Train the model based on samples"""

        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        for _ in range(self.c.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.c.mini_batch_size):
                # get mini batch
                end = start + self.c.mini_batch_size
                mini_batch_indexes = indexes[start:end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(clip_range = clip_range,
                                       samples = mini_batch)
                # compute gradients
                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 0.5)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 32)
                self.scaler.step(self.optimizer)
                self.scaler.update()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-4)

    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range: float) -> torch.Tensor:
        """## PPO Loss"""

        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples['values'] + samples['advantages']

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        pi, value = self.model(samples['obs'])

        # #### Policy

        # $-\log \pi_\theta (a_t|s_t)$, $a_t$ are actions sampled from $\pi_{\theta_{OLD}}$
        log_pi = pi.log_prob(samples['actions'])

        # ratio $r_t(\theta) = \frac{\pi_\theta (a_t|s_t)}{\pi_{\theta_{OLD}} (a_t|s_t)}$;
        # *this is different from rewards* $r_t$.
        ratio = torch.exp(log_pi - samples['log_pis'])

        # The ratio is clipped to be close to 1.
        # We take the minimum so that the gradient will only pull
        # $\pi_\theta$ towards $\pi_{\theta_{OLD}}$ if the ratio is
        # not between $1 - \epsilon$ and $1 + \epsilon$.
        # This keeps the KL divergence between $\pi_\theta$
        #  and $\pi_{\theta_{OLD}}$ constrained.
        # Large deviation can cause performance collapse;
        #  where the policy performance drops and doesn't recover because
        #  we are sampling from a bad policy.
        #
        # Using the normalized advantage
        #  $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$
        #  introduces a bias to the policy gradient estimator,
        #  but it reduces variance a lot.
        clipped_ratio = ratio.clamp(min = 1.0 - clip_range,
                                    max = 1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()

        # #### Entropy Bonus

        # $\mathcal{L}^{EB}(\theta) =
        #  \mathbb{E}\Bigl[ S\bigl[\pi_\theta\bigr] (s_t) \Bigr]$
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # #### Value

        # Clipping makes sure the value function $V_\theta$ doesn't deviate
        #  significantly from $V_{\theta_{OLD}}$.
        clipped_value = samples['values'] + (value - samples['values']).clamp(min = -clip_range,
                                                                              max = clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()

        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) -
        #  c_1 \mathcal{L}^{VF} (\theta) + c_2 \mathcal{L}^{EB}(\theta)$

        # we want to maximize $\mathcal{L}^{CLIP+VF+EB}(\theta)$
        # so we take the negative of it as the loss
        loss = -(policy_reward - self.c.vf_weight * vf_loss + self.c.entropy_weight * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()

        tracker.add({'policy_reward': policy_reward,
                     'vf_loss': vf_loss,
                     'entropy_bonus': entropy_bonus,
                     'kl_div': approx_kl_divergence,
                     'clip_fraction': clip_fraction})

        return loss

    def run_training_loop(self):
        """### Run training loop"""
        for update in monit.loop(self.c.updates):
            progress = update / self.c.updates
            # decreasing `learning_rate` and `clip_range` $\epsilon$
            learning_rate = self.c.start_lr * (1 - progress)
            clip_range = self.c.start_clipping_range * (1 - progress)
            # sample with current policy
            samples = self.sample()
            # train the model
            self.train(samples, learning_rate, clip_range)
            # write summary info to the writer, and log to the screen
            tracker.save()
            if (update + 1) % 50 == 0:
                logger.log()
            if (update + 1) % 500 == 0:
                configs = {i: self.c.__getattribute__(i) for i in self.c.__annotations__}
                torch.save((self.model, configs), '/tmp2/b06902021/tetris/%s_%06d.pkl' %
                        (experiment.get_uuid()[:8], update + 1))

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))

if __name__ == "__main__":
    conf = Configs()
    with experiment.record(
            name = 'Tetris_PPO_float16',
            exp_conf = conf):
        m = Main(conf)
        experiment.add_pytorch_models({'model': m.model})
        try: m.run_training_loop()
        except Exception as e: print(traceback.format_exc())
        m.destroy()
