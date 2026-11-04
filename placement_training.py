"""
The Agency, Reinforcement Learning for Tetris

DQN-style training for the placement-based agent (see placement.py).

One *decision* = choosing where to put one piece. For a placement a, with
afterstate features f(a) and reward r(a):

    V(f(a))  is trained towards   r(a)                                   if the placement ends the game
                                  r(a) + gamma * max_b V_target(f'(b))   otherwise

where b ranges over the legal placements of the *next* piece. This is the
same Bellman target as the per-tick DQN, but one decision spans a whole piece,
so credit assignment is immediate.

Run:  uv run python placement_training.py --decisions 150000 --save-dir run-placement
"""

import argparse
import os
import random
import time
from collections import deque, namedtuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from tetris_engine import GameState
from placement import (PlacementNet, PlacementReward, get_candidates,
                       best_candidate_index)

# feats: afterstate features of the placement taken; next_feats: features of
# every legal placement of the next piece (None if the game ended)
Transition = namedtuple("Transition", ("feats", "reward", "next_feats"))


class PlacementReplay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, feats, reward, next_feats):
        # half precision keeps the memory footprint small
        self.memory.append(Transition(
            torch.tensor(feats, dtype=torch.float16), reward,
            None if next_feats is None
            else torch.tensor(next_feats, dtype=torch.float16)))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PlacementTrainer:
    def __init__(self, save_dir="run-placement", reward_cfg=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.reward_cfg = reward_cfg or PlacementReward()

        self.total_decisions = 150_000
        self.gamma = 0.95
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.replay_capacity = 50_000
        self.replay_start_size = 1_000
        self.target_update_frequency = 500
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_decisions = 50_000
        self.max_placements_per_episode = 500
        self.print_frequency = 5_000
        self.checkpoint_frequency = 25_000
        self.eval_frequency = 10_000
        self.eval_games = 10
        self.eval_max_placements = 300

    def epsilon(self, decision):
        t = min(decision / self.epsilon_decay_decisions, 1.0)
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, pick, label):
        """Plays games with {pick(candidates) -> index}; prints a summary."""
        placements, lines, capped = [], [], 0
        for _ in range(self.eval_games):
            game = GameState()
            n = 0
            while not game.stop and n < self.eval_max_placements:
                candidates = get_candidates(game, self.reward_cfg)
                game.place_piece(candidates[pick(candidates)].cells)
                n += 1
            placements.append(n)
            lines.append(game.lines_cleared_total)
            capped += n >= self.eval_max_placements
        print(f"[eval] {label}: {self.eval_games} games, "
              f"mean pieces placed {sum(placements) / len(placements):.1f}, "
              f"mean lines {sum(lines) / len(lines):.1f}, "
              f"max lines {max(lines)}, hit cap {capped}")

    def evaluate_greedy(self, net):
        net.eval()
        self.evaluate(lambda c: best_candidate_index(net, c), "greedy policy")
        net.train()

    # -- training -----------------------------------------------------------
    def train_step(self, net, target_net, optimizer, replay):
        batch = replay.sample(self.batch_size)
        feats = torch.stack([t.feats for t in batch]).float()
        targets = torch.tensor([t.reward for t in batch], dtype=torch.float32)

        live = [i for i, t in enumerate(batch) if t.next_feats is not None]
        if live:
            nexts = [batch[i].next_feats for i in live]
            sizes = [n.shape[0] for n in nexts]
            with torch.no_grad():
                values = target_net(torch.cat(nexts).float())
                best = torch.stack([v.max() for v in torch.split(values, sizes)])
            targets[torch.tensor(live)] += self.gamma * best

        loss = F.smooth_l1_loss(net(feats), targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()
        return loss.item()

    def train(self):
        net = PlacementNet()
        target_net = PlacementNet()
        target_net.load_state_dict(net.state_dict())
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        replay = PlacementReplay(self.replay_capacity)

        self.evaluate(lambda c: random.randrange(len(c)), "random placement baseline")
        print()

        game = GameState()
        pending = None  # (feats, reward) of the previous decision
        placements_this_game = 0
        episodes = lines_total = pieces_total = 0
        loss_total, loss_steps = 0.0, 0
        last_time = time.time()

        for decision in range(1, self.total_decisions + 1):
            candidates = get_candidates(game, self.reward_cfg)
            if pending is not None:
                replay.push(pending[0], pending[1], [c.feats for c in candidates])
                pending = None
            if placements_this_game >= self.max_placements_per_episode:
                # truncated, not game over: the transition above was stored
                # with real next-state candidates, so just start a new game
                episodes += 1
                lines_total += game.lines_cleared_total
                pieces_total += placements_this_game
                game, placements_this_game = GameState(), 0
                candidates = get_candidates(game, self.reward_cfg)

            eps = self.epsilon(decision)
            if random.random() < eps:
                choice = random.choice(candidates)
            else:
                with torch.no_grad():
                    choice = candidates[best_candidate_index(net, candidates)]

            game.place_piece(choice.cells)
            placements_this_game += 1

            if choice.done or game.stop:
                replay.push(choice.feats, choice.reward, None)
                episodes += 1
                lines_total += game.lines_cleared_total
                pieces_total += placements_this_game
                game, placements_this_game = GameState(), 0
            else:
                pending = (choice.feats, choice.reward)

            if len(replay) >= self.replay_start_size:
                loss_total += self.train_step(net, target_net, optimizer, replay)
                loss_steps += 1
            if decision % self.target_update_frequency == 0:
                target_net.load_state_dict(net.state_dict())

            if decision % self.print_frequency == 0:
                now = time.time()
                n = max(episodes, 1)
                print(f"decision {decision} ({now - last_time:.1f}s since last print)")
                print(f"  epsilon {eps:.3f}, episodes {episodes}")
                print(f"  avg pieces per episode {pieces_total / n:.1f}, "
                      f"avg lines per episode {lines_total / n:.2f}")
                print(f"  avg loss {loss_total / loss_steps:.4f}" if loss_steps
                      else "  avg loss N/A")
                print()
                episodes = lines_total = pieces_total = 0
                loss_total, loss_steps = 0.0, 0
                last_time = now
            if decision % self.eval_frequency == 0:
                self.evaluate_greedy(net)
                print()
            if decision % self.checkpoint_frequency == 0:
                path = os.path.join(self.save_dir, "placement-%06d.pt" % decision)
                torch.save(net.state_dict(), path)
                print(f"saved {path}\n")

        torch.save(net.state_dict(), os.path.join(self.save_dir, "placement-final.pt"))
        return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=150_000)
    parser.add_argument("--save-dir", default="run-placement")
    args = parser.parse_args()
    trainer = PlacementTrainer(save_dir=args.save_dir)
    trainer.total_decisions = args.decisions
    trainer.train()
