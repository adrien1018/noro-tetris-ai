from labml.configs import BaseConfigs

class Configs(BaseConfigs):
    # #### Configurations
    # $\gamma$ and $\lambda$ for advantage calculation
    gamma: float = 0.998
    lamda: float = 0.95
    # number of updates
    updates: int = 70000
    # number of epochs to train the model with sampled data
    epochs: int = 2
    # number of worker processes
    n_workers: int = 4
    env_per_worker: int = 16
    # number of steps to run on each process for a single update
    worker_steps: int = 128
    # size of mini batches
    mini_batch_size: int = 1024
    channels: int = 128
    blocks: int = 8
    lr: float = 1e-4
    clipping_range: float = 0.2
    vf_weight: float = 0.5
    entropy_weight: float = 1e-2
    reg_l2: float = 2e-4
