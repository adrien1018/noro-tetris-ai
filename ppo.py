import multiprocessing, sys, hashlib
import multiprocessing.connection
from typing import Dict, List

import tetris
import numpy as np, torch
from labml import monit, tracker, logger, experiment
from labml.configs import BaseConfigs
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F

device = torch.device('cuda')
# board, valid, current(7), next(7), speed(14)
kInChannel = 1 + 1 + 7 + 7 + 14
kH, kW = 20, 10
kTensorDim = (3, kH, kW)

class Game:
    def __init__(self, seed: int):
        self.args = (0, False)
        self.env = tetris.Tetris(*self.args)
        self.env.Seed(seed)
        # board, current(7), next(7), speed(14)
        self.obs = np.zeros(kTensorDim)
        # keep track of the episode rewards
        self.rewards = []
        self.cnt = 0

    def set_obs(self):
        self.obs[:] = 0
        self.obs[0] = 1 - self.env.board
        self.obs[1] = self.env.allowed[0]
        self.obs[2,0,0] = self.env.cur
        self.obs[2,0,1] = self.env.nxt
        self.obs[2,0,2] = self.env.Speed()

    def step(self, action):
        """
        Executes `action` (x, y)
         returns a tuple of (observation, reward, done, info).
        """
        if self.env.over or self.cnt >= 40: return self.obs, 0, True, None
        suc, score, _ = self.env.Place(*action)
        reward = score + 0.01 if suc else 0
        self.cnt = 0 if suc else self.cnt + 1
        self.set_obs()
        # maintain rewards for each step
        self.rewards.append(reward)
        over = self.env.over or self.cnt >= 40

        if over:
            # if finished, set episode information if episode is over, and reset
            episode_info = {"reward": sum(self.rewards), "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None
        return self.obs, reward, over, episode_info

    def reset(self):
        """
        Reset environment
        """
        self.env.Reset(*self.args)
        self.set_obs()
        self.rewards = []
        self.cnt = 0
        return self.obs

def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    """
    Each worker process runs this method
    """
    # create game
    game = Game(seed)
    # wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step((data // kW, data % kW)))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker:
    """
    Creates a new worker and runs it in a separate process.
    """
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()

class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                )
        self.final = nn.ReLU(True)
    def forward(self, x):
        return self.final(self.main(x) + x)

class Model(nn.Module):
    def __init__(self, ch, blk):
        super().__init__()
        self.start = nn.Sequential(
                nn.Conv2d(kInChannel, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                )
        self.linear = nn.Linear(kH * kW, kH * kW)
        self.res = nn.Sequential(*[ConvBlock(ch) for i in range(blk)])
        # A fully connected layer to get logits for $\pi$
        self.pi_logits_head = nn.Sequential(
                nn.Conv2d(ch, 4, 1),
                nn.BatchNorm2d(4),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(4 * kH * kW, kH * kW)
                )
        # A fully connected layer to get value function
        self.value = nn.Sequential(
                nn.Conv2d(ch, 1, 1),
                nn.BatchNorm2d(1),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(1 * kH * kW, 256),
                nn.ReLU(True),
                nn.Linear(256, 1)
                )

    def forward(self, obs: torch.Tensor):
        q = torch.zeros((obs.shape[0], kInChannel, kH, kW), dtype = torch.float32, device = device)
        q[:,0:2] = obs[:,0:2]
        q.scatter_(1, (2 + obs[:,2,0,0].type(torch.long)).view(-1, 1, 1, 1).repeat(1, 1, kH, kW), 1)
        q.scatter_(1, (9 + obs[:,2,0,1].type(torch.long)).view(-1, 1, 1, 1).repeat(1, 1, kH, kW), 1)
        q.scatter_(1, (16 + obs[:,2,0,2].type(torch.long)).view(-1, 1, 1, 1).repeat(1, 1, kH, kW), 1)
        x = self.start(q)
        x = self.res(x)
        pi = self.pi_logits_head(x) + self.linear(q[:,1:2].reshape(-1, kH * kW))
        value = self.value(x).reshape(-1)
        pi_sample = Categorical(logits = torch.clamp(pi, -30, 30))
        return pi_sample, value

def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.int32, device = device)

class Configs(BaseConfigs):
    # $\gamma$ and $\lambda$ for advantage calculation
    gamma: float = 0.99
    lamda: float = 0.95
    # number of updates
    updates: int = 200000
    # number of epochs to train the model with sampled data
    epochs: int = 1
    # number of worker processes
    n_workers: int = 4
    # number of steps to run on each process for a single update
    worker_steps: int = 128
    # number of mini batches
    n_mini_batch: int = 16
    channels: int = 128
    blocks: int = 8
    start_lr: float = 3e-5
    vf_weight: float = 0.5
    entropy_weight: float = 1e-2

class Main:
    def __init__(self, c: Configs):
        # #### Configurations

        self.c = c
        # total number of samples for a single update
        self.batch_size = self.c.n_workers * self.c.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.c.batch_size // self.c.n_mini_batch
        assert (self.batch_size % self.c.n_mini_batch == 0)

        # #### Initialize

        # create workers
        self.workers = [Worker(47 + i) for i in range(self.c.n_workers)]

        # initialize tensors for observations
        self.obs = np.zeros((self.c.n_workers, *kTensorDim))
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        # model for sampling
        self.model = Model(c.channels, c.blocks).to(device)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.c.start_lr)

    def sample(self) -> (Dict[str, np.ndarray], List):
        """### Sample data with current policy"""

        rewards = np.zeros((self.c.n_workers, self.c.worker_steps), dtype = np.float32)
        actions = np.zeros((self.c.n_workers, self.c.worker_steps), dtype = np.int32)
        done = np.zeros((self.c.n_workers, self.c.worker_steps), dtype = np.bool)
        obs = np.zeros((self.c.n_workers, self.c.worker_steps, *kTensorDim), dtype = np.uint8)
        log_pis = np.zeros((self.c.n_workers, self.c.worker_steps), dtype = np.float32)
        values = np.zeros((self.c.n_workers, self.c.worker_steps), dtype = np.float32)

        # sample `worker_steps` from each worker
        for t in range(self.c.worker_steps):
            with torch.no_grad():
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `n_workers`
                pi, v = self.model(obs_to_torch(self.obs))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

            # run sampled actions on each worker
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))

            for w, worker in enumerate(self.workers):
                # get results after executing the actions
                self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()

                # collect episode info, which is available if an episode finished;
                #  this includes total reward and length of the episode -
                #  look at `Game` to see how it works.
                # We also add a game frame to it for monitoring.
                if info:
                    tracker.add('reward', info['reward'])
                    tracker.add('length', info['length'])
                    tracker.add('ratio', info['reward'] / info['length'] * 100)

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
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        ### Calculate advantages
        \begin{align}
        \hat{A_t^{(1)}} &= r_t + \gamma V(s_{t+1}) - V(s)
        \\
        \hat{A_t^{(2)}} &= r_t + \gamma r_{t+1} +\gamma^2 V(s_{t+2}) - V(s)
        \\
        ...
        \\
        \hat{A_t^{(\infty)}} &= r_t + \gamma r_{t+1} +\gamma^2 r_{t+1} + ... - V(s)
        \end{align}
        $\hat{A_t^{(1)}}$ is high bias, low variance whilst
        $\hat{A_t^{(\infty)}}$ is unbiased, high variance.
        We take a weighted average of $\hat{A_t^{(k)}}$ to balance bias and variance.
        This is called Generalized Advantage Estimation.
        $$\hat{A_t} = \hat{A_t^{GAE}} = \sum_k w_k \hat{A_t^{(k)}}$$
        We set $w_k = \lambda^{k-1}$, this gives clean calculation for
        $\hat{A_t}$
        \begin{align}
        \delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t)$
        \\
        \hat{A_t} &= \delta_t + \gamma \lambda \delta_{t+1} + ... +
                             (\gamma \lambda)^{T - t + 1} \delta_{T - 1}$
        \\
        &= \delta_t + \gamma \lambda \hat{A_{t+1}}
        \end{align}
        """

        # advantages table
        advantages = np.zeros((self.c.n_workers, self.c.worker_steps), dtype = np.float32)
        last_advantage = 0

        # $V(s_{t+1})$
        _, last_value = self.model(obs_to_torch(self.obs))
        last_value = last_value.cpu().data.numpy()

        for t in reversed(range(self.c.worker_steps)):
            # mask if episode completed after step $t$
            mask = 1.0 - done[:, t]
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
        """
        ### Train the model based on samples
        """

        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        for _ in range(self.c.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(clip_range=clip_range,
                                       samples=mini_batch)

                # compute gradients
                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate
                self.optimizer.zero_grad()
                loss.backward()
                for x in self.model.parameters():
                    x.grad[torch.isnan(x.grad)] = 0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 0.5)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 2)
                for x in self.model.parameters():
                    x.grad[torch.isnan(x.grad)] = 0
                for x in self.model.parameters():
                    if torch.isnan(x).any(): print('a')
                self.optimizer.step()
                for x in self.model.parameters():
                    if torch.isnan(x).any(): print('b')


    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-4)

    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range: float) -> torch.Tensor:
        """
        ## PPO Loss
        We want to maximize policy reward
         $$\max_\theta J(\pi_\theta) =
           \mathop{\mathbb{E}}_{\tau \sim \pi_\theta}\Biggl[\sum_{t=0}^\infty \gamma^t r_t \Biggr]$$
         where $r$ is the reward, $\pi$ is the policy, $\tau$ is a trajectory sampled from policy,
         and $\gamma$ is the discount factor between $[0, 1]$.
        \begin{align}
        \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
         \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
        \Biggr] &=
        \\
        \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
          \sum_{t=0}^\infty \gamma^t \Bigl(
           Q^{\pi_{OLD}}(s_t, a_t) - V^{\pi_{OLD}}(s_t)
          \Bigr)
         \Biggr] &=
        \\
        \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
          \sum_{t=0}^\infty \gamma^t \Bigl(
           r_t + V^{\pi_{OLD}}(s_{t+1}) - V^{\pi_{OLD}}(s_t)
          \Bigr)
         \Biggr] &=
        \\
        \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
          \sum_{t=0}^\infty \gamma^t \Bigl(
           r_t
          \Bigr)
         \Biggr]
         - \mathbb{E}_{\tau \sim \pi_\theta}
            \Biggl[V^{\pi_{OLD}}(s_0)\Biggr] &=
        J(\pi_\theta) - J(\pi_{\theta_{OLD}})
        \end{align}
        So,
         $$\max_\theta J(\pi_\theta) =
           \max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
              \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
           \Biggr]$$
        Define discounted-future state distribution,
         $$d^\pi(s) = (1 - \gamma) \sum_{t=0}^\infty \gamma^t P(s_t = s | \pi)$$
        Then,
        \begin{align}
        J(\pi_\theta) - J(\pi_{\theta_{OLD}})
        &= \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
         \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
        \Biggr]
        \\
        &= \frac{1}{1 - \gamma}
         \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \Bigl[
          A^{\pi_{OLD}}(s, a)
         \Bigr]
        \end{align}
        Importance sampling $a$ from $\pi_{\theta_{OLD}}$,
        \begin{align}
        J(\pi_\theta) - J(\pi_{\theta_{OLD}})
        &= \frac{1}{1 - \gamma}
         \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \Bigl[
          A^{\pi_{OLD}}(s, a)
         \Bigr]
        \\
        &= \frac{1}{1 - \gamma}
         \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_{\theta_{OLD}}} \Biggl[
          \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
         \Biggr]
        \end{align}
        Then we assume $d^\pi_\theta(s)$ and  $d^\pi_{\theta_{OLD}}(s)$ are similar.
        The error we introduce to $J(\pi_\theta) - J(\pi_{\theta_{OLD}})$
         by this assumtion is bound by the KL divergence between
         $\pi_\theta$ and $\pi_{\theta_{OLD}}$.
        [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
         shows the proof of this. I haven't read it.
        \begin{align}
        J(\pi_\theta) - J(\pi_{\theta_{OLD}})
        &= \frac{1}{1 - \gamma}
         \mathop{\mathbb{E}}_{s \sim d^{\pi_\theta} \atop a \sim \pi_{\theta_{OLD}}} \Biggl[
          \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
         \Biggr]
        \\
        &\approx \frac{1}{1 - \gamma}
         \mathop{\mathbb{E}}_{\color{orange}{s \sim d^{\pi_{\theta_{OLD}}}}
         \atop a \sim \pi_{\theta_{OLD}}} \Biggl[
          \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
         \Biggr]
        \\
        &= \frac{1}{1 - \gamma} \mathcal{L}^{CPI}
        \end{align}
        """

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

        # \begin{align}
        # \mathcal{L}^{CLIP}(\theta) =
        #  \mathbb{E}_{a_t, s_t \sim \pi_{\theta{OLD}}} \biggl[
        #    min \Bigl(r_t(\theta) \bar{A_t},
        #              clip \bigl(
        #               r_t(\theta), 1 - \epsilon, 1 + \epsilon
        #              \bigr) \bar{A_t}
        #    \Bigr)
        #  \biggr]
        # \end{align}
        #
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

        # \begin{align}
        # V^{\pi_\theta}_{CLIP}(s_t)
        #  &= clip\Bigl(V^{\pi_\theta}(s_t) - \hat{V_t}, -\epsilon, +\epsilon\Bigr)
        # \\
        # \mathcal{L}^{VF}(\theta)
        #  &= \frac{1}{2} \mathbb{E} \biggl[
        #   max\Bigl(\bigl(V^{\pi_\theta}(s_t) - R_t\bigr)^2,
        #       \bigl(V^{\pi_\theta}_{CLIP}(s_t) - R_t\bigr)^2\Bigr)
        #  \biggr]
        # \end{align}
        #
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
        """
        ### Run training loop
        """
        for update in monit.loop(self.c.updates):
            progress = update / self.c.updates
            # decreasing `learning_rate` and `clip_range` $\epsilon$
            learning_rate = self.c.start_lr * (1 - progress)
            clip_range = 0.1 * (1 - progress)
            # sample with current policy
            samples = self.sample()
            # train the model
            self.train(samples, learning_rate, clip_range)
            # write summary info to the writer, and log to the screen
            tracker.save()
            if (update + 1) % 50 == 0:
                logger.log()
            if (update + 1) % 200 == 0:
                configs = {i: self.config.__getattribute__(i) for i in self.config.__annotations__}
                torch.save((self.model, configs), '/tmp2/b06902021/tetris/%s_%06d.pkl' %
                        (experiment.get_uuid()[:8], update + 1))

    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))

if __name__ == "__main__":
    conf = Configs()
    with experiment.record(
            name = 'Tetris PPO 2',
            exp_conf = conf):
        m = Main(conf)
        experiment.add_pytorch_models({'model': m.model})
        try: m.run_training_loop()
        except Exception as e: print(e)
        m.destroy()
