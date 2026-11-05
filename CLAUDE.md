# CLAUDE.md

## What this repo is
A from-scratch Tetris engine plus a Deep Q-Network (DQN) trainer, built by "The Agency" (GT). An agent learns to play Tetris from a 10x20 board using hyperparameters from the DeepMind DQN paper. Python + PyTorch + pygame. There is no package structure, build system or tests: every file is a flat script in the repo root.

## Files
| File | Role |
|---|---|
| `tetris_engine.py` | Game simulation. `GameState(width=10, height=20)`, `Action` enum (IDLE, LEFT, RIGHT, ROTATE_CW, ROTATE_CCW, RESET), 7 tetrominoes. `update(action)` advances one tick (gravity included) and returns lines cleared. `stop` is True on game over. `GameState(seed=n)` gives the game its own random generator, so the piece sequence is reproducible (used for fair evaluation); without a seed it uses the global `random` module. |
| `tetris_ui.py` | Pygame UI. Play by keyboard (arrows/WASD, `R` resets) or watch an agent. |
| `game_agent.py` | Agents with `get_move(state)`: `RandomAgent`, `BruteAgent` / `BruteAgent2` (brute-force lookahead), `ModelAgent` (epsilon-greedy over a torch Q-network). |
| `model.py` | `Net`: MLP 400 -> 100 -> 25 -> 5 (Q-value per action), leaky ReLU. |
| `placeholder_pytorch_model.py` | `PlaceholderModel`, a single linear layer, default for `ModelAgent`. |
| `tetris_utils.py` | `convert_gamestate_to_tensor` (400-dim: 200 locked cells + 200 current-piece cells) and `ReplayMemory`/`Transition`. |
| `reward_functions.py` | Reward shaping: `LinesClearedReward`, `LinesClearedMultiplierReward`, `HeightPenaltyReward(multiplier, game_over_penalty)`, `multipleRewards` (height/holes/bumpiness/line-clear weights). All expose `update_and_get_reward(state, action)`. |
| `training_loop.py` | DQN training: replay memory (200k), batch 32, gamma 0.99, RMSprop lr 2.5e-4, target network synced every 10k frames, epsilon 1.0 -> 0.1 over 1M frames, 10M frames total. Checkpoints (`checkpoint-NNN.pt`, whole model via `torch.save`) every 100k frames. |
| `board_metrics.py` | Column heights, holes, bumpiness, and the board-quality potential shared by rewards and the placement agent. |
| `placement.py` | **Placement-based agent.** `get_candidates(state)` enumerates every legal final placement of the current piece (BFS over the engine's own dynamics, so paths are exactly executable) with its afterstate features and reward. `PlacementNet` scores the *position* after a placement and a placement is chosen by `exact reward + gamma * value` (checkpoints are format 2; older bare-state-dict checkpoints still load and are scored the old way); `PlacementAgent` picks the best and, via `get_move`, steers the piece with normal per-tick actions so it works in `tetris_ui.py`. |
| `placement_training.py` | DQN training for the placement agent (one decision per piece; Huber loss, target net, replay). Prints a random baseline, periodic greedy `[eval]` lines, and saves `placement-NNNNNN.pt` (state_dicts). |
| `evaluate.py` | Offline evaluation of placement checkpoints: seeded greedy games (default 30 games, 1,000-piece cap), reports deaths, pieces, lines (95% CI) and end-of-game holes/height, optional random baseline (`--baseline`). Slow (~35 ms per placement), so run it in a second terminal while training. |
| `tetris_reward.py` | Headless benchmark (SDL dummy driver) comparing `BruteAgent` and `RandomAgent` mean rewards. |
| `engine_cli.py` | DEPRECATED text-mode play. Use the UI instead. |
| `game.py` | Empty stub. |

## How to use
Requires Python 3.10+ with `torch`, `pygame` and `numpy`, managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`). Setup: `uv sync`. Run scripts with `uv run <script>`, e.g. `uv run python tetris_ui.py`.

- **Play or watch:** `uv run python tetris_ui.py`. Edit the script to choose who plays. Set `IS_KEYBOARD_MODE = True` for human play. Near the bottom, `agent = ...` selects the agent (`RandomAgent()` is active; commented lines show how to load a checkpoint with `ModelAgent(torch.load("path.pt"))` or use `BruteAgent2`).
- **Train:** `uv run python training_loop.py`. The last lines of the file construct `TrainingLoop(save_dir=..., reward_func=...)` and call `loop.loop()` under `__main__`, so edit the save dir and reward function there. Training stops at `training_frames` (10M). `epsilon_schedule` (list of `(frame, epsilon)` points, linearly interpolated) defaults to the DeepMind 1.0 -> 0.1 over 1M frames. Progress prints every 10k frames (incl. average lines cleared per episode, from `GameState.lines_cleared_total`). At start it prints a random-agent baseline, and every `eval_frequency` (100k) frames it plays `eval_games` (20) greedy games (epsilon 0) and prints mean length / lines. Compare that `[eval] greedy policy` line to the baseline: it is a better progress signal than the training averages, which are dominated by epsilon-random moves.
- **Train the placement agent (recommended, learns far faster):** `uv run python placement_training.py --decisions 150000 --save-dir run-placement` (about 23 ms per decision). After ~1.5k decisions the greedy policy already clears ~14 lines/game vs 0 for random placement. Watch it in the UI by enabling the `PlacementAgent(load_placement_model(...))` line in `tetris_ui.py`.
- **Evaluate checkpoints properly:** `uv run python evaluate.py run-placement/placement-100000.pt run-placement/placement-final.pt --baseline`. The in-training `[eval]` line is deliberately light (10 games, 500-piece cap, every 25k decisions; change with `--eval-games/--eval-cap/--eval-every`) and noisy; `evaluate.py` uses the same fixed seeds for every checkpoint so results are comparable.
- **Benchmark agents:** `uv run python tetris_reward.py`.

## Gotchas
- Per-tick DQN (`training_loop.py`) learns slowly because each piece takes ~18 decisions to place; the placement agent avoids that. The per-tick path now uses `BoardDeltaReward`, Huber loss and gradient clipping (an earlier run with large absolute penalties and squared error diverged: Q-values far outside any possible return).
- Engine quirk: game over is decided by the *first* cell that can't fall, so a piece can lock with a cell in the top two rows without ending the game; the next spawn then overwrites that locked cell.
- Board is `game_board[x][y]` with origin at bottom-left. Cell values: 0 empty, +k locked piece type k, -k the current falling piece.
- `BruteAgent` and `tetris_reward.py` call `state.get_reward()`, which `GameState` doesn't define (rewards moved to `reward_functions.py`). Use `BruteAgent2` with a reward function instead.
- `tetris_ui.py` instantiates `ModelAgent()` (untrained placeholder) and then overwrites it with `RandomAgent()`. The module-level script code runs on import.
- `utils`' docstring on output order is stale. The actual action order is the `Action` enum values 0-4.
- `Net` outputs 5 Q-values (no RESET). The training loop resets by constructing a new `GameState`, with episodes capped at 1000 steps.
- Checkpoint dirs (e.g. `run-4-12/`) are written to the working directory.
