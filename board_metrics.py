"""
The Agency, Reinforcement Learning for Tetris

Board-quality measurements (column heights, holes, bumpiness) shared by the
reward functions and the placement-based agent. Pure functions on a board in
the {GameState.game_board} layout: board[x][y], y = 0 is the bottom, values
> 0 are locked cells. Anything <= 0 (empty or the falling piece) is ignored.
"""

# Weights from the Tetris heuristic literature (Yiyuan Lee's write-up).
HOLE_WEIGHT = 0.36
BUMPINESS_WEIGHT = 0.18


def board_metrics(board, width, height):
    """Returns (column heights, holes per column, |height differences|)."""
    heights = [0] * width
    holes = [0] * width
    for x in range(width):
        col = board[x]
        top = 0
        for y in range(height - 1, -1, -1):
            if col[y] > 0:
                top = y + 1
                break
        heights[x] = top
        holes[x] = sum(1 for y in range(top) if col[y] == 0)
    bumpiness = [abs(heights[i] - heights[i + 1]) for i in range(width - 1)]
    return heights, holes, bumpiness


def potential_from_metrics(holes, bumpiness):
    """Higher (closer to 0) is a cleaner board."""
    return -(HOLE_WEIGHT * sum(holes) + BUMPINESS_WEIGHT * sum(bumpiness))


def potential(board, width, height):
    _, holes, bumpiness = board_metrics(board, width, height)
    return potential_from_metrics(holes, bumpiness)


def metrics_to_features(heights, holes, bumpiness, width, height):
    """Normalized feature vector (length 3 * width + 3) for a neural net."""
    features = [h / height for h in heights]
    features += [c / height for c in holes]
    features += [b / height for b in bumpiness]
    features.append(sum(heights) / (width * height))
    features.append(sum(holes) / (width * height) * 4)
    features.append(max(heights) / height)
    return features
