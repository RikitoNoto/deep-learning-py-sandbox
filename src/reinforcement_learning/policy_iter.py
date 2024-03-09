if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from environment import Actions, GridWorld
from policy_eval import policy_eval


def argmax(d: dict):
    """d (dict)"""
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V: dict, env: GridWorld, gamma: float):
    pi = {}

    # 全ての状態で全ての行動の行動価値を計算し、
    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]  # r(s,a,s') + γvμ(s')
            action_values[action] = value

        # 評価が一番高い行動を1.0(100%)で、他の行動を0(0%)にする。
        max_action = argmax(action_values)
        action_probs = {
            Actions.UP: 0.0,
            Actions.DOWN: 0.0,
            Actions.LEFT: 0.0,
            Actions.RIGHT: 0.0,
        }
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def policy_iter(env: GridWorld, gamma: float, threshold=0.001, is_render=True):
    pi = defaultdict(
        lambda: {
            Actions.UP: 0.25,
            Actions.DOWN: 0.25,
            Actions.LEFT: 0.25,
            Actions.RIGHT: 0.25,
        }
    )
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break
        pi = new_pi

    return pi


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma)
