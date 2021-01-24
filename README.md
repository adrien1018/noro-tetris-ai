# No-Rotation Tetris AI

A no-rotation Tetris AI trained by [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) algorithm.

Here's [a demonstration](https://www.youtube.com/watch?v=jjbwDUGDiPo) using this AI.

## Requirements

First, run the following command to build the library for core game logic (written in C++):
```
make -C tetris
```

Python requirements are listed in `requirements.txt`. Install them by
```
pip install -r requirements.txt
```

## Usage

### Run the AI on Jstris

Open Jstris website, and fill in the screen coordinates of the playfield to the 8th and 10th line of `run.py`. After that, run

```
./run.py
```

and then **immediately** switch to the Jstris window. The program will capture the game field and play the game automatically (by simulating keyboard inputs).

### Training a model

```
./train.py [--some_hyperparameter=value ...] experiment_name [run_uuid]
```
The program use [LabML](https://github.com/lab-ml/labml) library to monitor and save the training status. It will save the model every 500 iterations. `run_uuid` is used to resume an interrupted training.

For the list of available hyperparameters, run `./train.py --help` or refer to `config.py`.

The training part is only tested on Linux environments. LabML seems to fail on Windows systems.

The main training code is modified from [vpj/rl_samples](https://github.com/vpj/rl_samples).

### Evaluating

```
./evaluate.py [seed]
```

The program plays 10000 games using the trained model and outputs the scoring distribution. The format of each line of output is `[lines] [number of games]`.

The table below shows the approximated cumulative distribution of lines:

| Lines | Probability |
| ----- | ----------- |
| 50+   | 97.3%       |
| 75+   | 86.4%       |
| 100+  | 69.5%       |
| 125+  | 50.1%       |
| 150+  | 33.0%       |
| 175+  | 19.2%       |
| 200+  | 9.6%        |
| 225+  | 3.8%        |
| 250+  | 1.1%        |
| 275+  | 0.3%        |
| 300+  | 0.1%        |

### Examining the policy

```
./view.py
```

The program will simulate a game, and display the following information alternately each time you hit the Enter key:

- The current piece and the next piece
- The placement made by the model

This can be useful to examine or learn the model's policy.

## Model details

### Network architecture

The input is a $21 \times 10 \times 61$ image stack comprising of 61 feature planes:
- 1 feature plane for the current playfield (1 if empty, 0 otherwise)
- 1 feature plane for the valid placements (1 if valid, 0 otherwise)
- One-hot encoding of the current piece (7 feature planes)
- One-hot encoding of each of the next 5 pieces (35 feature planes)
- One-hot encoding of the current piece count modulo 7 (7 feature planes)
- One-hot encoding of the holded piece (8 feature planes)
- 1 feature plane for whether hold is used
- 1 feature plane for the current score (divided by 32)

The input features are processed by a single convolutional block followed by 8 residual blocks, all of them having 128 filters. The output of the residual tower is passed into two separate heads for computing the value and policy.

### Reward

Rewards are given according to the table below:

| Condition         | Reward       |
| ----------------- | ------------ |
| 0 - 105 lines     | 0.5 per line |
| 105 - 120 lines   | 3 per line   |
| 120+ lines        | 8 per line   |
| Invalid placement | -0.125 per placement<br>Game ends after 3 consecutive invalid placements |

### Training hyperparameters

The model in the repository is trained by the given default hyperparameters, except those listed in the table below:

| Iterations      | lr   | reg_l2 |
| --------------- | ---- | ------ |
| 0 - 30000       | 1e-4 | 5e-5   |
| 30000 - 50000   | 1e-5 | 0      |
| 50000 - 65000   | 5e-6 | 0      |
