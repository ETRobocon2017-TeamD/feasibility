import enum


class Action(enum.Enum):
    u"""エージェントが取りうる行動"""
    # ↑後ろに全力
    ACTION1 = 0
    ACTION2 = 1
    ACTION3 = 2
    ACTION4 = 3
    ACTION5 = 4
    # ↓前に全力


def get_reward(inputs):
    """環境の値から報酬を計算する"""
    x, _, angle, _ = inputs
    # loss = abs(x + 1) ** 2 + abs(angle * 5 + 1) ** 2
    loss = 1 + abs(angle)
    # loss = abs(x + 1) ** 2 + abs(angle * 5 + 1) ** 2
    return -loss
