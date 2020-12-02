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

### Run the AI with FCEUX

Open FCEUX and load the ROM first, and adjust the screen size to 512 x 448 (the second smallest size). **Before** entering the main game screen, run:

```
./run.py models/model.pth
```

After the program outputs `Ready`, start the game. The program will capture the FCEUX screen continually to parse the game state, and outputs something like this after any piece is spawned:

```
[[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 1]
 [0 0 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 1 1 1 1 1]
 [0 0 0 0 0 0 0 1 1 1]
 [0 0 0 1 1 1 1 1 1 1]
 [0 0 0 1 1 1 1 1 1 1]
 [0 0 0 1 1 1 1 1 1 1]
 [2 2 0 1 1 1 1 1 1 1]
 [2 2 0 0 0 1 1 0 1 1]
 [1 1 1 0 1 1 1 1 1 0]
 [0 0 0 0 0 1 1 1 1 0]
 [0 1 0 0 1 1 1 1 0 1]
 [0 1 1 0 0 1 1 0 1 1]
 [1 1 0 1 1 1 1 1 1 1]
 [0 1 1 1 0 0 1 1 1 1]
 [0 0 1 1 1 1 1 1 1 1]
 [0 0 0 0 1 1 1 1 1 1]
 [0 0 0 1 1 1 1 1 1 1]
 [0 1 1 1 1 0 1 1 1 0]]
0:LLLL
8 1
[next] 22/32 -> 0:L 3:LLL          [next] 27/32 -> 0:LLLL
[next]  5/32 -> 0:RRRR             [next]  5/32 -> 0:LLLL 3:R
[next]  5/32 -> 0:L 3:LL
```

- The first part indicates the current playfield (`0`/`1`) and the desired placement of the current piece (`2`).
- After the playfield is a line indicating the input sequence to make the desired placement.
    - Format: `[row number]:[L/Rs]`.
    - Rows are numbered 0 to 19 from top to bottom.
    - A prefix `---` indicates no input needed at the start of drop (0-th row).
- The next line is the coordinate of the desired placement.
- The next few lines are the input sequences corresponding to possible desired placements of the next piece.
  - Format: `[probability] -> [input sequence]`.
  - Use the first column if one places the current piece at the indicated position;
    use the second column if one places the current piece by the highest probability placement before the next piece information is available (that is, the input sequence displayed at the first row of the first column when the previous piece is dropping).

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
./evaluate.py models/model.pth [seed]
```

The program plays 10000 games using the trained model and outputs the scoring distribution. The format of each line of output is `[lines] [number of games]`.

The table below shows the approximated cumulative distribution of lines:

| Lines | Probability |
| ----- | ----------- |
| 5+    | 99.8%       |
| 10+   | 97.7%       |
| 15+   | 89.4%       |
| 20+   | 71.6%       |
| 25+   | 47.8%       |
| 30+   | 26.3%       |
| 35+   | 12.1%       |
| 40+   | 4.6%        |
| 45+   | 1.5%        |
| 50+   | 0.4%        |
| 55+   | 0.1%        |

### Examining the policy

```
./view.py models/model.pth [seed]
```

The program will simulate a game, and display the following information alternately each time you hit the Enter key:

- The current piece and the next piece
- The placement made by the model

This can be useful to examine or learn the model's policy.

## Model details

### Network architecture

The input is a $20 \times 10 \times 17$ image stack comprising of 17 feature planes:
- 1 feature plane for the current playfield (1 if empty, 0 otherwise)
- 1 feature plane for the valid placements (1 if valid, 0 otherwise)
- One-hot encoding of the current piece (7 feature planes)
- One-hot encoding of the next piece (7 feature planes)
- 1 feature plane for the current score (divided by 32)

The input features are processed by a single convolutional block followed by 8 residual blocks, all of them having 128 filters. The output of the residual tower is passed into two separate heads for computing the value and policy.

### Reward

Rewards are given according to the table below:

| Condition         | Reward       |
| ----------------- | ------------ |
| 0 - 29 lines      | 0.5 per line |
| 30 - 34 lines     | 3 per line   |
| 35+ lines         | 8 per line   |
| Invalid placement | -0.125 per placement<br>Game ends after 3 consecutive invalid placements |

### Training hyperparameters

The model in the repository is trained by the given default hyperparameters, except those listed in the table below:

| Iterations      | lr   | reg_l2 |
| --------------- | ---- | ------ |
| 0 - 45000       | 1e-4 | 2e-4   |
| 45000 - 70000   | 1e-4 | 5e-5   |
| 70000 - 90000   | 5e-5 | 0      |
| 90000 - 110000  | 2e-5 | 0      |
| 110000 - 130000 | 8e-6 | 0      |
