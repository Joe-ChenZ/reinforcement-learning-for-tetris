# CLAUDE.md

## What this repo is
A from-scratch Tetris engine plus a Deep Q-Network (DQN) trainer, built by "The Agency" (GT). An agent learns to play Tetris from a 10x20 board using hyperparameters from the DeepMind DQN paper. Python + PyTorch + pygame. There is no package structure, build system or tests: every file is a flat script in the repo root.

## Files
| File | Role |
|---|---|
| `tetris_engine.py` | Game simulation. `GameState(width=10, height=20)`, `Action` enum (IDLE, LEFT, RIGHT, ROTATE_CW, ROTATE_CCW, RESET), 7 tetrominoes. `update(action)` advances one tick (gravity included) and returns lines cleared. `stop` is True on game over. |
| `tetris_ui.py` | Pygame UI. Play by keyboard (arrows/WASD, `R` resets) or watch an agent. |
| `game_agent.py` | Agents with `get_move(state)`: `RandomAgent`, `BruteAgent` / `BruteAgent2` (brute-force lookahead), `ModelAgent` (epsilon-greedy over a torch Q-network). |
| `model.py` | `Net`: MLP 400 -> 100 -> 25 -> 5 (Q-value per action), leaky ReLU. |
| `placeholder_pytorch_model.py` | `PlaceholderModel`, a single linear layer, default for `ModelAgent`. |
| `tetris_utils.py` | `convert_gamestate_to_tensor` (400-dim: 200 locked cells + 200 current-piece cells) and `ReplayMemory`/`Transition`. |
| `reward_functions.py` | Reward shaping: `LinesClearedReward`, `LinesClearedMultiplierReward`, `HeightPenaltyReward(multiplier, game_over_penalty)`, `multipleRewards` (height/holes/bumpiness/line-clear weights). All expose `update_and_get_reward(state, action)`. |
| `training_loop.py` | DQN training: replay memory (200k), batch 32, gamma 0.99, RMSprop lr 2.5e-4, target network synced every 10k frames, epsilon 1.0 -> 0.1 over 1M frames, 10M frames total. Checkpoints (`checkpoint-NNN.pt`, whole model via `torch.save`) every 100k frames. |
| `tetris_reward.py` | Headless benchmark (SDL dummy driver) comparing `BruteAgent` and `RandomAgent` mean rewards. |
| `engine_cli.py` | DEPRECATED text-mode play. Use the UI instead. |
| `game.py` | Empty stub. |

## How to use
Requires Python 3 with `torch` and `pygame`. Setup: `python -m venv .venv`, activate it (`.venv\Scriptsactivate` on Windows), then `pip install -r requirements.txt`.

- **Play or watch:** `python tetris_ui.py`. Edit the script to choose who plays. Set `IS_KEYBOARD_MODE = True` for human play. Near the bottom, `agent = ...` selects the agent (`RandomAgent()` is active; commented lines show how to load a checkpoint with `ModelAgent(torch.load("path.pt"))` or use `BruteAgent2`).
- **Train:** `python training_loop.py`. The last lines of the file construct `TrainingLoop(save_dir=..., reward_func=...)` and call `loop.loop(...)` at import time, so edit the save dir and reward function there. The argument to `loop()` is ignored; training stops at `training_frames` (10M). Progress prints every 10k frames.
- **Benchmark agents:** `python tetris_reward.py`.

## Gotchas
- Board is `game_board[x][y]` with origin at bottom-left. Cell values: 0 empty, +k locked piece type k, -k the current falling piece.
- `training_loop.py` has `from audioop import mul`, an unused import. `audioop` was removed in Python 3.13, so delete that line on new Pythons.
- `training_loop.py` ignores `epsilon_schedule`. `determine_epsilon_from_schedule` is hardcoded to the DeepMind decay.
- `BruteAgent` and `tetris_reward.py` call `state.get_reward()`, which `GameState` doesn't define (rewards moved to `reward_functions.py`). Use `BruteAgent2` with a reward function instead.
- `tetris_ui.py` instantiates `ModelAgent()` (untrained placeholder) and then overwrites it with `RandomAgent()`. The module-level script code runs on import.
- `utils`' docstring on output order is stale. The actual action order is the `Action` enum values 0-4.
- `Net` outputs 5 Q-values (no RESET). The training loop resets by constructing a new `GameState`, with episodes capped at 1000 steps.
- Checkpoint dirs (e.g. `run-4-12/`) are written to the working directory.
