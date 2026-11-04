"""
Authors: Neil Thistlethwaite
(add your name above if you contribute to this file)
The Agency, Reinforcement Learning for Tetris

Allows us to do "reward shaping" and provide smoother rewards to our agent
rather than just when it clears lines.
"""



## These constants aren't used right now, they were from the reward refactor.

GAME_END_REWARD = -200

# Applied when a line is cleared
LINE_CLEAR_REWARD = 10

# Applied to disincentivize higher stacks, so if current stack is 3 high,
# adds reward 3 * (PER_HIGHEST_PIECE_REWARD). Should be negative.
PER_HIGHEST_PIECE_REWARD = -5

# Applied for each "isolated hole" present in the stack so far.
PER_ISOLATED_HOLE_REWARD = -3

# Applied whenever a new piece is received
NEW_PIECE_REWARD = -2


from board_metrics import potential


class RewardFunction:
    def __init__(self):
        pass

    def update_and_get_reward(self, state, action):
        """Calls {state.update} with {action} and returns the reward."""
        line_clear_reward = state.update(action)
        end_game_penalty = -50000 * state.stop
        # print(end_game_penalty)
        return line_clear_reward + end_game_penalty


class LinesClearedReward(RewardFunction):
    """The base class effectively already does this."""
    pass


class LinesClearedMultiplierReward(RewardFunction):
    def update_and_get_reward(self, state, action):
        lines_cleared = state.update(action)
        if lines_cleared == 0:
            return 0
        elif lines_cleared == 1:
            return 1
        elif lines_cleared == 2:
            return 3
        elif lines_cleared == 3:
            return 6
        elif lines_cleared == 4:
            return 10

class HeightPenaltyReward(RewardFunction):
    def __init__(self, multiplier, game_over_penalty):
        self.multiplier = multiplier
        self.game_over_penalty = game_over_penalty
    
    def update_and_get_reward(self, state, action):
        lines_cleared = state.update(action)
        # will add a penalty between [0, -multiplier]
        # depending on how tall the stack is
        highest_row = 0
        for row in range(state.height-1, -1, -1):
            if any(state.game_board[i][row] > 0 for i in range(state.width)):
                highest_row = row + 1
                break
        reward = lines_cleared - self.multiplier * (highest_row / state.height)

        if state.stop:
            reward = reward - self.game_over_penalty
        return reward

        end_game_penalty = -10 * state.stop
        # print(end_game_penalty)
        return reward + end_game_penalty

class BoardDeltaReward(RewardFunction):
    """
    Rewards line clears plus the *change* in board quality (holes and
    bumpiness, see board_metrics.py) caused by the step, minus a game-over
    penalty. Unlike the absolute per-frame penalties in {multipleRewards}:
      - rewards are order 1 (no -500 cliffs), and
      - they are ~0 while a piece is falling and show up when it locks,
      - there is no incentive to end the game early to stop paying a penalty.
    """
    def __init__(self, line_rewards=(0, 1, 3, 6, 10), shaping=1.0,
                 game_over_penalty=5.0):
        self.line_rewards = line_rewards
        self.shaping = shaping
        self.game_over_penalty = game_over_penalty

    def update_and_get_reward(self, state, action):
        before = potential(state.game_board, state.width, state.height)
        lines_cleared = state.update(action)
        after = potential(state.game_board, state.width, state.height)
        reward = self.line_rewards[lines_cleared] + self.shaping * (after - before)
        if state.stop:
            reward -= self.game_over_penalty
        return reward


# http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
class multipleRewards(RewardFunction):
    def __init__(self, height_mult=-0.67, hole_mult=-0.36, bumpiness_mult=-0.18, lineclear_mult=0.8,
                 game_over_penalty=50000, reward_scale=1.0):
        self.height_mult = height_mult
        self.hole_mult = hole_mult
        self.lineclear_mult = lineclear_mult
        self.bumpiness_mult = bumpiness_mult
        self.game_over_penalty = game_over_penalty
        # multiplies the final reward; the raw values are huge (tens per frame,
        # -50000 on death) which is hard for a neural net to regress, so scale
        # them down (e.g. 0.01) while keeping the ratios between the terms
        self.reward_scale = reward_scale
    def update_and_get_reward(self, state, action):
        # NOTE: {state.update} must be called exactly once per call. This used
        # to call it twice (once via the parent class, once here), which
        # advanced the game two ticks and applied the action twice, so the
        # replay-memory transitions didn't match the action that was stored.
        lines_cleared = state.update(action)
        parent_reward = lines_cleared - self.game_over_penalty * state.stop
        lines_cleared_reward = lines_cleared ** 2 * self.lineclear_mult
        height_penalty = 0
        hole_penalty = 0
        global_max_height = 0
        top_row = 0
        max_height_cols = []
        bumpiness_penalty = 0
        for i in range(state.width):
            max_height = -1
            j = 0
            holes = 0
            while j < state.height:
                if state.game_board[i][j] > 0:
                    max_height = j
                    # holes += 1
                
                # if state.game_board[i][j] < 0:
                #     piece_height_sum += j
                j += 1
            
            for k in range(0, max_height):
                if state.game_board[i][k] == 0:
                    holes += 1
            max_height += 1
            max_height_cols.append(max_height)

            hole_penalty += holes
            global_max_height = max(global_max_height, max_height)
            
            # height_penalty += (max_height + 1) * self.height_mult
            # top_row = max(top_row, max_height)
            # hole_penalty += sum(state.game_board[i][h] == 0 for h in range(0, max_height))
            # if hole_count > max_height:
            #     print(state.get_current_board())
            # print("max height and hole count: ", max_height, hole_count)
        # print(max_height_cols)
        for i in range(0, state.width, 2):
            left = i - 1
            right = i + 1
            if left >= 0:
                bumpiness_penalty += abs(max_height_cols[i] - max_height_cols[left])
                # if abs(max_height_cols[i] - max_height_cols[left]):
                    # print("at col ", i ,", the bumpiness =", abs(max_height_cols[i] - max_height_cols[left]))
            if right < state.width:
                bumpiness_penalty += abs(max_height_cols[i] - max_height_cols[right])
                # if abs(max_height_cols[i] - max_height_cols[right]):
                #     print("at col ", i ,", the bumpiness =", abs(max_height_cols[i] - max_height_cols[right]))
            
        
        # print(state.get_current_board())

        height_penalty = global_max_height * 10
        bumpiness_penalty *= self.bumpiness_mult
        height_penalty *= self.height_mult
        hole_penalty *= self.hole_mult
        # print("bumpiness = ", bumpiness_penalty)
        # print("height = ", height_penalty)
        # print("hole = ", hole_penalty)
        # print("----------------------------------")
        # print("hole penalty:", hole_penalty)
        # print("height penalty", height_penalty)
        # height_penalty = highest_row * self.height_mult * 20

        # height_penalty *= self.height_mult
        # for j in range(top_row):
        #     for i in range(state.width):
        #         hole_penalty += state.game_board[i][j] == 0
        # hole_penalty *= self.hole_mult
        return self.reward_scale * (parent_reward + height_penalty + hole_penalty + bumpiness_penalty + lines_cleared_reward)


### old code from state refactor: consider using later?
##    def get_reward(self):
##        reward = self.reward
##        for row in range(self.height-1, -1, -1):
##            if any(self.game_board[i][row] > 0 for i in range(self.width)):
##                # something is on row i, apply penalty
##                reward += PER_HIGHEST_PIECE_REWARD * (row + 1)
##                break
##        for i in range(1, self.width - 1):
##            for j in range(1, self.height - 1):
##                if self.game_board[i][j] == 0:
##                    if all(self.game_board[i+dx][j+dy] > 0 for dx, dy in
##                           ((1,0),(0,1),(-1,0),(0,-1))):
##                        reward += PER_ISOLATED_HOLE_REWARD
##        return reward
