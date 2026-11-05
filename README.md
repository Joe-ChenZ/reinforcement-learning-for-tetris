# Reinforcement Learning for Tetris

A from-scratch Tetris engine and two deep Q-learning agents that learn to play it, built by The Agency (GT). Python, PyTorch and pygame.

- **Placement agent (recommended).** Once per piece it chooses *where the piece ends up* and learns from the result. It clears about 148 lines per 400 pieces after 30k training decisions (about 12 minutes of training); the maximum is 160, and a random placement agent clears none.
- **Per-tick DQN.** The original approach: one left/right/rotate decision per frame, using the DeepMind DQN setup. It learns far more slowly and is kept for comparison.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## Quick start

**Train the placement agent**

```
uv run python placement_training.py --decisions 150000 --save-dir run-placement
```

It prints a random-placement baseline first, then training stats every 5k decisions and a greedy evaluation (no exploration) every 10k. Checkpoints are written to the save directory every 25k decisions, plus `placement-final.pt` at the end. Training runs at roughly 25 ms per decision.

**Watch an agent play**

```
uv run python tetris_ui.py
```

Edit the `agent = ...` line near the bottom of `tetris_ui.py` to choose who plays, for example:

```python
agent = PlacementAgent(load_placement_model("run-placement/placement-final.pt"))
```

Set `IS_KEYBOARD_MODE = True` to play yourself (arrow keys or WASD, `R` to restart).

**Evaluate checkpoints**

```
uv run python evaluate.py run-placement/placement-final.pt --baseline
```

Plays 30 seeded games (1,000-piece cap) per checkpoint and reports deaths, pieces, lines and end-of-game holes. It is slow, so run it in a second terminal while training continues. The `[eval]` lines printed during training are a lighter, noisier version of the same thing.

**Train the per-tick DQN**

```
uv run python training_loop.py
```

Edit the `TrainingLoop(...)` line at the bottom of the file to set the save directory and reward function.

## How it works

### The placement agent
1. **Enumerate options.** For the current piece, a breadth-first search over the engine's own rules (a move, then gravity) finds every place it can lock, together with the exact actions that get it there (`placement.py`).
2. **Describe each option.** Each placement is simulated, and the resulting board is turned into 34 numbers: column heights, holes, bumpiness, totals, lines cleared and a game-over flag (`board_metrics.py`).
3. **Score each option.** `score = exact reward + gamma * W(board)`. The reward is computed directly (line clears, change in holes and bumpiness, game-over penalty). `W` is a small neural network (`PlacementNet`) that estimates how good the resulting board is for the future.
4. **Choose** the highest score, or a random placement with probability epsilon while training.
5. **Learn.** After each piece, `W` is trained towards `max over the next piece's placements of (reward + gamma * W)`, using the standard DQN ingredients: experience replay, a periodically synced target network, Huber loss, gradient clipping and an epsilon schedule (`placement_training.py`).

Why this design:
- **One decision per piece** removes the credit-assignment problem of per-frame control, where a piece takes about 18 decisions to place.
- **Learning the future value only** (and adding the exact reward when choosing) stops the network's approximation error from hiding the difference between, say, clearing a line now and clearing it later.
- **Engineered features** give the network the quantities that decide Tetris quality, so it does not have to rediscover them from raw cells.

### The per-tick DQN
The network sees the raw 400-cell board (locked cells plus the falling piece) and outputs a Q-value for each of five actions. It uses a replay buffer, a target network, Huber loss and the `BoardDeltaReward` reward. Hyperparameters follow the DeepMind paper.

## Results and known limits

- Placement agent, 30k decisions, 400-piece games: 147.8 lines on average, a skipped-clean-clear rate of 9%, about 2 holes on the board on average.
- Over very long games it eventually fails. In a 500-piece test, 3 to 5 of 20 games died, usually after 380 or more pieces, with 16 or more buried holes. Holes accumulate because they are only penalized when created. Ideas for improving this are listed in the notes below.
- The per-tick DQN barely clears lines after more than a million frames.

Notes:
- Evaluation games in the training log stop at 300 pieces, so small dips there are mostly one or two games dying by chance (10 games per evaluation). Use `evaluate.py` (more games, longer cap) to compare checkpoints.
- Possible improvements: a higher episode cap and lower final epsilon so training sees late-game boards, hole-depth and row-transition features, an ongoing penalty for existing holes, and a decaying learning rate.
- Engine quirk: game over is decided by the first piece cell that cannot fall, so a piece can lock with a cell in the top two rows without ending the game; the next spawn then overwrites that cell.

## Repository layout

| File | Purpose |
|---|---|
| `tetris_engine.py` | Game simulation (`GameState`, `Action`, pieces, line clears) |
| `tetris_ui.py` | Pygame viewer / keyboard play |
| `placement.py` | Placement enumeration, features, `PlacementNet`, `PlacementAgent` |
| `placement_training.py` | Training for the placement agent |
| `evaluate.py` | Offline, seeded evaluation of placement checkpoints |
| `board_metrics.py` | Heights, holes and bumpiness shared by rewards and features |
| `training_loop.py` | Per-tick DQN training |
| `model.py` | Per-tick `Net` (400 → 100 → 25 → 5) |
| `reward_functions.py` | Reward functions for the per-tick path |
| `game_agent.py` | Agents: random, brute-force lookahead, per-tick model agent |
| `tetris_utils.py` | Board-to-tensor conversion and replay memory for the per-tick path |

`engine_cli.py`, `game.py`, `tetris_reward.py` and `placeholder_pytorch_model.py` are older or auxiliary files. See `CLAUDE.md` for more detailed notes on each file and known gotchas.
