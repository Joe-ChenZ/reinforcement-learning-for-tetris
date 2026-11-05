# Placement agent: results report

A reference for what the placement agent is, how it was trained and evaluated, and what the numbers say. Written after the first full 150k-decision run.

## Summary

- The placement agent (one decision per piece) learns to play Tetris well. The final model of a 150k-decision run cleared **396.5 of a possible 400 lines** in 1,000-piece games and died in only **1 of 30** games.
- A random placement agent dies after about 21 pieces and clears no lines.
- Quality changes a lot within a run: the 75k checkpoint died in **15 of 30** games, the 100k checkpoint in 2 of 30. Among the three checkpoints evaluated, quality only improved, but whether the later checkpoints (125k, 150k) stay as good was not checked.
- The per-tick DQN (one left/right/rotate decision per frame) barely cleared lines after more than a million frames, which is why the placement agent exists.

## The agent

- **Decision:** once per piece, choose where it ends up. A breadth-first search over the engine's own rules (move, then gravity) lists every legal final placement and the actions that reach it (`placement.py`).
- **Score of a placement:** `exact reward + gamma * W(board after the placement)`. `W` is `PlacementNet`, an MLP 34 -> 128 -> 64 -> 1 that estimates the future value of a board. It never has to learn the immediate reward, which is computed directly.
- **Features (34):** 10 column heights, 10 per-column hole counts, 9 bumpiness values, 3 totals, lines cleared, game-over flag (`board_metrics.py`).
- **Reward:** `line_rewards = (0, 1, 3, 6, 10)` for 0 to 4 lines, plus a shaping term equal to the change in `-(hole_weight * holes + bumpiness_weight * bumpiness)` (`hole_weight = 1.0`, `bumpiness_weight = 0.2`), or `-5` if the placement ends the game.
- **Training (`placement_training.py`):** DQN with experience replay (50,000), a target network synced every 500 decisions, Huber loss, gradient-norm clipping at 10, Adam with learning rate 1e-3, batch 64, gamma 0.95, epsilon 1.0 -> 0.05 over 50,000 decisions, and 500-piece training episodes. The target for a board is `max over the next piece's placements of (reward + gamma * W_target(next board))`, or 0 if the game ended.

## How it was evaluated

`evaluate.py`, with the greedy policy (no exploration), 30 games per checkpoint, seeds 5000 to 5029, and a cap of 1,000 pieces. Every checkpoint and the random baseline play exactly the same piece sequences (`GameState(seed=n)` gives each game its own random generator). A game that reaches the cap counts as survived. The most lines possible in 1,000 pieces is 400 (4 cells per piece on a 10-wide board).

```
uv run python evaluate.py models/placement-150k/placement-075000.pt models/placement-150k/placement-100000.pt models/placement-150k/placement-final.pt --baseline
```

Intervals are 95% confidence intervals over the 30 games.

## Results

| Agent | Died before 1,000 pieces | Pieces placed | Lines cleared | Holes at end | Max height at end |
|---|---|---|---|---|---|
| Random placement | 30/30 (about 21 pieces) | 20.7 ± 1.2 | 0.0 ± 0.1 | 60.6 | 19.7 |
| `placement-075000` | **15/30** | 770.6 ± 111.2 | 299.3 ± 46.0 | 9.9 | 12.6 |
| `placement-100000` | 2/30 (at pieces 595, 670) | 975.5 ± 33.6 | 386.5 ± 14.5 | 2.5 | 6.1 |
| `placement-final` (150k) | 1/30 (at piece 978) | 999.3 ± 1.4 | 396.5 ± 1.2 | 1.7 | 5.7 |

## What the results show

1. **The approach works.** Greedy play with a learned value is close to the ceiling on this test: the final model clears 99% of the possible lines and ends with under two holes on average.
2. **Quality changes fast within a run.** The 75k checkpoint loses half its games, with about 10 holes and a stack 12.6 high at the end. 25,000 decisions later the same run loses 2 of 30. Among the evaluated checkpoints the order is simply 75k < 100k < final, so these results do not show whether training ever gets worse. The unevaluated 125k and 150k checkpoints are where that would show up.
3. **Failures are slow.** The games that died did so late (pieces 595 to 978 for the good checkpoints), after holes had built up. A short evaluation cannot see this. The earlier in-training evaluation (10 games, 300-piece cap) could not.
4. **Small differences need many games.** The 75k checkpoint's interval (770.6 ± 111.2 pieces) is wide, and with 10 games it would have been wider.

## Caveats

- **One training run.** Results from a single seed. RL results vary between runs, so the ranking of checkpoints, and the dip at 75k, may not repeat.
- **Only three checkpoints evaluated.** The 25k, 50k, 125k and 150k checkpoints are saved in `models/placement-150k/` but were not evaluated, so there is no complete learning curve.
- **The 1,000-piece cap hides slower failure.** The final model might still die in longer games. Only one of its 30 games died, but the cap stops the test at 1,000 pieces.
- **Evaluation uses the same 30 seeds for every checkpoint.** That is the point (fair comparison), but anything tuned against these seeds would need a separate held-out set.
- **Timing is for one machine.** Greedy play runs at about 35 ms per placement, so a full evaluation of one checkpoint takes 10 to 20 minutes.

## Next steps

- Evaluate every checkpoint, and keep the best one by a larger evaluation instead of the last one, if later ones turn out to dip.
- Evaluate the remaining checkpoints for a full learning curve.
- Make training more stable and reduce late-game hole build-up: learning-rate decay, a higher training episode cap and lower final epsilon so training sees late-game boards, an ongoing penalty for existing holes, and hole-depth and row-transition features.
- Repeat the run with more than one training seed before comparing settings.

## Files

| Path | Contents |
|---|---|
| `models/placement-150k/` | All seven checkpoints of the 150k run (25k to 150k, plus `placement-final.pt`), 54 KB each, format 2 (`{"format": 2, "state_dict": ...}`). Load with `placement.load_placement_model(path)`. |
| `evaluate.py` | The evaluation used for the table above |
| `placement.py`, `placement_training.py`, `board_metrics.py` | Agent, training and features |
