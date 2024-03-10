import numpy as np
import matplotlib.pyplot as plt


def argmax(xs: dict):
    """
    受けとった辞書から最大のvalueを持つキーを返す。
    キーが全て同じであったり、最大のvalueを持つキーが複数ある場合は
    その中からランダムに返す
    """
    idxes = {key: value for key, value in xs.items() if value == max(xs.values())}
    # idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return list(idxes.keys())[0]
    elif len(idxes) == 0:

        return list(xs.keys())[np.random.randint(0, len(xs.keys()))]

    selected = list(idxes.keys())[np.random.randint(0, len(idxes.keys()))]
    return selected


def greedy_probs(Q, state, actions: list, epsilon=0):
    qs = {action: Q[(state, action)] for action in actions}
    max_action = argmax(qs)  # OR np.argmax(qs)
    base_prob = epsilon / len(actions)
    action_probs = {
        action: base_prob for action in actions
    }  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += 1 - epsilon
    return action_probs


def plot_total_reward(reward_history):
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()
