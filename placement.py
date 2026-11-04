"""
The Agency, Reinforcement Learning for Tetris

Placement-based play. Instead of choosing one primitive move per tick, the
agent chooses, once per piece, *where the piece ends up* (a rotation + column).
For every legal final placement we build the resulting board (the
"afterstate"), and a small value network scores it:

    V(afterstate) ~ reward for making that placement + discounted future value

The agent picks the placement with the highest value. See
{placement_training.py} for training and PlacementAgent for playing (it also
works in tetris_ui.py: it converts the chosen placement into the normal
per-tick actions).
"""

import random
from collections import deque, namedtuple

import torch
import torch.nn as nn

from tetris_engine import Action
from game_agent import Agent
from board_metrics import (board_metrics, metrics_to_features,
                           potential_from_metrics)

ACTIONS = [Action.IDLE, Action.LEFT, Action.RIGHT, Action.ROTATE_CW, Action.ROTATE_CCW]

# cells: final (x, y) cells of the piece; path: per-tick actions that steer
# the piece there from where it is now; feats: afterstate feature vector;
# reward: reward for making this placement; done: placement ends the game.
Candidate = namedtuple("Candidate",
                       ("cells", "path", "feats", "reward", "done", "lines"))


class PlacementReward:
    """Reward for one placement: line clears + change in board quality."""
    def __init__(self, line_rewards=(0, 1, 3, 6, 10), shaping=1.0,
                 game_over_penalty=5.0):
        self.line_rewards = line_rewards
        self.shaping = shaping
        self.game_over_penalty = game_over_penalty


# -- enumerating placements ---------------------------------------------------

def _is_free(board, width, height, x, y):
    # the falling piece's own cells are stored as negative numbers, so
    # "<= 0" matches the engine's {_is_valid_piece_location}
    return 0 <= x < width and 0 <= y < height and board[x][y] <= 0


def _apply_move(board, width, height, cells, shape_num, action):
    """The engine's move/rotate rules on a list of cells; None if blocked."""
    if action == Action.LEFT or action == Action.RIGHT:
        dx = 1 if action == Action.RIGHT else -1
        new_cells = [(x + dx, y) for x, y in cells]
    else:
        if shape_num == 2:  # the O piece doesn't rotate
            return None
        m = -1 if action == Action.ROTATE_CCW else 1
        cx, cy = cells[0]  # the engine rotates around the first cell
        new_cells = [(cx + (y - cy) * m, cy - (x - cx) * m) for x, y in cells]
    if all(_is_free(board, width, height, x, y) for x, y in new_cells):
        return new_cells
    return None


def _tick(board, width, height, cells, shape_num, action):
    """
    One engine tick on a list of cells: apply {action} if it is possible,
    then gravity. Returns (cells, locked); when the piece can't fall it locks
    where it is.
    """
    if action != Action.IDLE:
        moved = _apply_move(board, width, height, cells, shape_num, action)
        if moved is not None:
            cells = moved
    fallen = [(x, y - 1) for x, y in cells]
    if all(_is_free(board, width, height, x, y) for x, y in fallen):
        return fallen, False
    return cells, True


def legal_placements(state):
    """
    Returns a list of (final_cells, path) for every place the current piece
    can lock. Breadth-first search over the engine's own dynamics (every
    action is followed by one step of gravity), so the paths are exactly
    executable and "wait" (IDLE) steps appear where needed, e.g. to fall far
    enough to rotate. Paths are shortest-first; the last action locks.
    """
    board, width, height = state.game_board, state.width, state.height
    shape_num = state.game_piece.shape_num
    start = tuple(state.current_piece)
    parent = {start: None}
    queue = deque([start])
    placements = {}
    while queue:
        cells = queue.popleft()
        for action in ACTIONS:
            new_cells, locked = _tick(board, width, height, list(cells),
                                      shape_num, action)
            if locked:
                cell_set = frozenset(new_cells)
                if cell_set not in placements:
                    path = [action]
                    node = cells
                    while parent[node] is not None:
                        node, a = parent[node]
                        path.append(a)
                    path.reverse()
                    placements[cell_set] = (new_cells, path)
                continue
            key = tuple(new_cells)
            if key not in parent:
                parent[key] = (cells, action)
                queue.append(key)
    return list(placements.values())


# -- afterstates ---------------------------------------------------------------

def _clear_full_rows(board, width, height):
    full = [y for y in range(height - 1)
            if all(board[x][y] > 0 for x in range(width))]
    if not full:
        return board, 0
    full_set = set(full)
    cleared = [[v for y, v in enumerate(col) if y not in full_set]
               + [0] * len(full) for col in board]
    return cleared, len(full)


def _ends_game(locked_board, cells, width, height):
    """Mirrors the engine: the first cell that can't fall decides."""
    cell_set = set(cells)
    for x, y in cells:
        below_free = y - 1 >= 0 and (locked_board[x][y - 1] == 0
                                     or (x, y - 1) in cell_set)
        if not below_free:
            return y >= height - 2
    return False


def get_candidates(state, reward_cfg=None):
    """One Candidate for every legal placement of the current piece."""
    cfg = reward_cfg or PlacementReward()
    width, height = state.width, state.height
    # the locked board without the falling piece
    locked = [[c if c > 0 else 0 for c in col] for col in state.game_board]
    _, holes0, bump0 = board_metrics(locked, width, height)
    phi_before = potential_from_metrics(holes0, bump0)
    piece_value = state.game_piece.shape_num + 1

    candidates = []
    for cells, path in legal_placements(state):
        done = _ends_game(locked, cells, width, height)
        board = [col[:] for col in locked]
        for x, y in cells:
            board[x][y] = piece_value
        board, lines = _clear_full_rows(board, width, height)
        heights, holes, bump = board_metrics(board, width, height)
        if done:
            reward = -cfg.game_over_penalty
        else:
            phi_after = potential_from_metrics(holes, bump)
            reward = (cfg.line_rewards[lines]
                      + cfg.shaping * (phi_after - phi_before))
        feats = metrics_to_features(heights, holes, bump, width, height)
        feats += [lines / 4.0, 1.0 if done else 0.0]
        candidates.append(Candidate(cells, path, feats, reward, done, lines))
    return candidates


# -- the value network and the agent --------------------------------------------

# 10 heights + 10 holes + 9 bumpiness + 3 totals + (lines cleared, game over)
FEATURE_SIZE = 10 + 10 + 9 + 3 + 2  # for the default 10x20 board


class PlacementNet(nn.Module):
    """Maps an afterstate feature vector to a single value."""
    def __init__(self, input_size=FEATURE_SIZE, hidden=(128, 64)):
        super().__init__()
        layers = []
        last = input_size
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_placement_model(path):
    model = PlacementNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def best_candidate_index(model, candidates):
    with torch.no_grad():
        values = model(torch.tensor([c.feats for c in candidates]))
    return int(torch.argmax(values))


class RandomPlacementAgent(Agent):
    """Baseline: a uniformly random legal placement, teleported in."""
    def choose(self, state):
        return random.choice(get_candidates(state))


class PlacementAgent(Agent):
    """
    Picks the placement with the highest value. {get_move} turns the choice
    into primitive actions so the agent also works in tetris_ui.py: it
    re-plans every tick from the piece's current position, which absorbs the
    piece falling while it is being steered.
    """
    def __init__(self, model=None, epsilon=0.0):
        self.model = model if model is not None else PlacementNet()
        self.epsilon = epsilon

    def choose(self, state):
        candidates = get_candidates(state)
        if random.random() < self.epsilon:
            return random.choice(candidates)
        return candidates[best_candidate_index(self.model, candidates)]

    def get_move(self, state):
        if state.stop:
            return Action.RESET
        choice = self.choose(state)
        return choice.path[0] if choice.path else Action.IDLE
