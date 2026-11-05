"""
The Agency, Reinforcement Learning for Tetris

Offline evaluation of placement-agent checkpoints. Plays many seeded games
with the greedy policy (no exploration), so every checkpoint, and the random
baseline, faces exactly the same piece sequences. Slow (about 35 ms per
placement), so run it in a second terminal while training continues:

    uv run python evaluate.py run-placement/placement-100000.pt run-placement/placement-final.pt
    uv run python evaluate.py run-placement/*.pt --games 50 --cap 1000 --baseline

Reports, per checkpoint: how many games died before the cap, mean pieces
placed and lines cleared (with a 95% confidence interval), and the state of
the board at the end (holes, max height).
"""

import argparse
import math
import random
import statistics
import sys
import time
from typing import Callable, NamedTuple

import torch

from tetris_engine import GameState
from board_metrics import board_metrics
from placement import (Candidate, PlacementReward, best_candidate_index,
                       get_candidates, load_placement_model)


class GameResult(NamedTuple):
    pieces: int
    lines: int
    died: bool
    holes: int       # buried holes on the locked board at the end
    max_height: int


def play_game(pick: Callable[[list[Candidate]], int], seed: int, cap: int,
              reward_cfg: PlacementReward) -> GameResult:
    game = GameState(seed=seed)
    pieces = 0
    while not game.stop and pieces < cap:
        candidates = get_candidates(game, reward_cfg)
        game.place_piece(candidates[pick(candidates)].cells)
        pieces += 1
    locked = [[c if c > 0 else 0 for c in col] for col in game.game_board]
    heights, holes, _ = board_metrics(locked, game.width, game.height)
    return GameResult(pieces, game.lines_cleared_total, game.stop,
                      sum(holes), max(heights))


def mean_ci(values: list[float]) -> str:
    """'mean ± half-width of the 95% confidence interval'."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.1f}"
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return f"{mean:.1f} ± {half:.1f}"


def summarize(label: str, results: list[GameResult], cap: int, seconds: float) -> None:
    deaths = [r.pieces for r in results if r.died]
    print(f"{label}")
    print(f"  games {len(results)}, cap {cap}, died {len(deaths)}/{len(results)}"
          + (f" (at pieces {sorted(deaths)})" if 0 < len(deaths) <= 12 else ""))
    print(f"  pieces placed {mean_ci([r.pieces for r in results])}")
    print(f"  lines cleared {mean_ci([r.lines for r in results])}")
    print(f"  at the end: holes {statistics.mean(r.holes for r in results):.1f}, "
          f"max height {statistics.mean(r.max_height for r in results):.1f}")
    print(f"  ({seconds:.0f}s)\n", flush=True)


def evaluate(label: str, pick: Callable[[list[Candidate]], int], games: int,
             cap: int, seed: int, reward_cfg: PlacementReward) -> None:
    start = time.time()
    results = []
    for i in range(games):
        results.append(play_game(pick, seed + i, cap, reward_cfg))
        print(f"\r  {label}: game {i + 1}/{games}", end="", file=sys.stderr, flush=True)
    print("\r" + " " * 70 + "\r", end="", file=sys.stderr)
    summarize(label, results, cap, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("checkpoints", nargs="*", help="placement checkpoint .pt files")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--cap", type=int, default=1_000,
                        help="max pieces per game (games reaching it count as survived)")
    parser.add_argument("--seed", type=int, default=5000,
                        help="game i uses seed (seed + i); keep it fixed to compare runs")
    parser.add_argument("--baseline", action="store_true",
                        help="also evaluate a random-placement agent")
    args = parser.parse_args()
    if not args.checkpoints and not args.baseline:
        parser.error("give at least one checkpoint, or --baseline")

    torch.set_num_threads(1)  # tiny net: threads only add overhead (and let you run several evals at once)
    reward_cfg = PlacementReward()

    if args.baseline:
        rng = random.Random(0)  # its own stream: repeatable, and independent of the engine's
        evaluate("random placement", lambda c: rng.randrange(len(c)),
                 args.games, args.cap, args.seed, reward_cfg)

    for path in args.checkpoints:
        net = load_placement_model(path)
        evaluate(path, lambda c, net=net: best_candidate_index(net, c),
                 args.games, args.cap, args.seed, reward_cfg)


if __name__ == "__main__":
    main()
