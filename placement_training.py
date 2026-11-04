"""
The Agency, Reinforcement Learning for Tetris

DQN-style training for the placement agent (see placement.py).

One *decision* = choosing where to put one piece. A placement a has exact,
known reward r(a) and leads to an afterstate board with features f(a). The
network W(f) is the value of a *position* (what is still to come), and a
placement is scored as

    score(a) = r(a) + gamma * W(f(a))          (0 future value if a ends the game)

Training moves W(f(a)) towards the best score available for the next piece:

    W(f(a))  <-  max over legal placements b of the next piece of  score(b)

(and towards 0 if a ended the game). The network never has to learn the
immediate reward, which we can compute exactly, so a clean line clear is not
lost in approximation error. One decision spans a whole piece, so credit
assignment is immediate.

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
                       best_candidate_index, save_placement_model,
                       DEFAULT_GAMMA)

# feats: afterstate features of the placement taken. next_*: features,
# rewards and game-over flags of every legal placement of the next piece
# (None if the game ended)
Transition = namedtuple("Transition",
                        ("feats", "next_feats", "next_rewards", "next_done"))


class PlacementReplay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, feats, next_candidates):
        """{next_candidates} is the next piece's candidate list, or None."""
        feats = torch.tensor(feats, dtype=torch.float16)  # half precision: small
        if next_candidates is None:
            self.memory.append(Transition(feats, None, None, None))
            return
        self.memory.append(Transition(
            feats,
            torch.tensor([c.feats for c in next_candidates], dtype=torch.float16),
            torch.tensor([c.reward for c in next_candidates], dtype=torch.float32),
            torch.tensor([1.0 if c.done else 0.0 for c in next_candidates])))

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
        self.gamma = DEFAULT_GAMMA
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
        self.evaluate(lambda c: best_candidate_index(net, c, self.gamma),
                      "greedy policy")
        net.train()

    # -- training -----------------------------------------------------------
    def train_step(self, net, target_net, optimizer, replay):
        batch = replay.sample(self.batch_size)
        feats = torch.stack([t.feats for t in batch]).float()
        targets = torch.zeros(self.batch_size)  # 0 where the game ended

        live = [i for i, t in enumerate(batch) if t.next_feats is not None]
        if live:
            sizes = [batch[i].next_feats.shape[0] for i in live]
            with torch.no_grad():
                values = target_net(torch.cat([batch[i].next_feats for i in live]).float())
                rewards = torch.cat([batch[i].next_rewards for i in live])
                done = torch.cat([batch[i].next_done for i in live])
                scores = rewards + self.gamma * (1.0 - done) * values
                best = torch.stack([v.max() for v in torch.split(scores, sizes)])
            targets[torch.tensor(live)] = best

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
        pending = None  # afterstate features of the previous decision
        placements_this_game = 0
        episodes = lines_total = pieces_total = 0
        loss_total, loss_steps = 0.0, 0
        last_time = time.time()

        for decision in range(1, self.total_decisions + 1):
            candidates = get_candidates(game, self.reward_cfg)
            if pending is not None:
                replay.push(pending, candidates)
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
                choice = candidates[best_candidate_index(net, candidates, self.gamma)]

            game.place_piece(choice.cells)
            placements_this_game += 1

            if choice.done or game.stop:
                replay.push(choice.feats, None)
                episodes += 1
                lines_total += game.lines_cleared_total
                pieces_total += placements_this_game
                game, placements_this_game = GameState(), 0
            else:
                pending = choice.feats

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
                save_placement_model(net, path)
                print(f"saved {path}\n")

        save_placement_model(net, os.path.join(self.save_dir, "placement-final.pt"))
        return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=150_000)
    parser.add_argument("--save-dir", default="run-placement")
    args = parser.parse_args()
    trainer = PlacementTrainer(save_dir=args.save_dir)
    trainer.total_decisions = args.decisions
    trainer.train()
