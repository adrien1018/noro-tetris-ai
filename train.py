#!/usr/bin/env python3

# Modified from https://github.com/vpj/rl_samples

import sys, traceback, os, subprocess, collections
from typing import Dict, List
from sortedcontainers import SortedList
import numpy as np, torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from labml import monit, tracker, logger, experiment

from game import Game, Worker, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.workers = [Worker(27 + i, c.env_per_worker) for i in range(self.c.n_workers)]
        self.score_queue = SortedQueue(400)

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
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.c.lr, weight_decay = self.c.reg_l2)

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
                actions_cpu = a.cpu().numpy()

            # run sampled actions on each worker
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions_cpu[self.w_range(w)]))

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
                    self.score_queue.add(info['score'])
                    tracker.add('reward', info['reward'])
                    tracker.add('score', info['score'])
                    tracker.add('score_per01', self.score_queue.get_ratio(0.01))
                    tracker.add('score_per10', self.score_queue.get_ratio(0.1))
                    tracker.add('score_per50', self.score_queue.get_ratio(0.5))
                    tracker.add('score_per90', self.score_queue.get_ratio(0.9))
                    tracker.add('score_per99', self.score_queue.get_ratio(0.99))
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

    def train(self, samples: Dict[str, torch.Tensor]):
        """### Train the model based on samples"""
        for _ in range(self.c.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)
            for start in range(0, self.batch_size, self.c.mini_batch_size):
                # get mini batch
                end = start + self.c.mini_batch_size
                mini_batch_indexes = indexes[start:end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]
                # train
                loss = self._calc_loss(clip_range = self.c.clipping_range,
                                       samples = mini_batch)
                # compute gradients
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
        sampled_return = (samples['values'] + samples['advantages']).float()
        sampled_normalized_advantage = self._normalize(samples['advantages'])
        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        pi, value = self.model(samples['obs'])

        # #### Policy
        log_pi = pi.log_prob(samples['actions'])
        # *this is different from rewards* $r_t$.
        ratio = torch.exp(log_pi - samples['log_pis'])
        # The ratio is clipped to be close to 1.
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
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # #### Value
        # Clipping makes sure the value function $V_\theta$ doesn't deviate
        #  significantly from $V_{\theta_{OLD}}$.
        clipped_value = samples['values'] + (value - samples['values']).clamp(
                min = -clip_range, max = clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()
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
        offset = tracker.get_global_step()
        tracker.set_queue('score', 400, True)
        for _ in monit.loop(self.c.updates - offset):
            update = tracker.get_global_step()
            progress = update / self.c.updates
            # sample with current policy
            samples = self.sample()
            # train the model
            self.train(samples)
            # write summary info to the writer, and log to the screen
            tracker.save()
            if (update + 1) % 25 == 0: logger.log()
            if (update + 1) % 500 == 0: experiment.save_checkpoint()

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('uuid', nargs = '?', default = '')
    conf = Configs()
    keys = conf._to_json()
    for key in keys:
        parser.add_argument('--' + key.replace('_', '-'), type = type(conf.__getattribute__(key)))
    args = vars(parser.parse_args())
    override_dict = {}
    for key in keys:
        if args[key] is not None: override_dict[key] = args[key]
    try:
        if len(args['name']) == 32:
            int(args['name'], 16)
            parser.error('Experiment name should not be uuid-like')
    except ValueError: pass
    experiment.create(name = args['name'])
    conf = Configs()
    experiment.configs(conf, override_dict)
    m = Main(conf)
    experiment.add_pytorch_models({'model': m.model})
    if len(args['uuid']): experiment.load(args['uuid'])
    with experiment.start():
        try: m.run_training_loop()
        except Exception as e: print(traceback.format_exc())
        m.destroy()
