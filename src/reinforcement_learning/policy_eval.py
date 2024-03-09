if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from reinforcement_learning.environment import Actions, GridWorld


def eval_one_step(
    pi: defaultdict[tuple[int, int], dict[Actions, float]],
    V: defaultdict,
    env: GridWorld,
    gamma=0.9,
):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(
    pi: defaultdict[tuple[int, int], dict[Actions, float]],
    V: defaultdict,
    env: GridWorld,
    gamma: float,
    threshold=0.001,
):
    while True:
        old_V = V.copy()
        V = eval_one_step(pi, V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(
        lambda: {
            Actions.UP: 0.6,
            Actions.DOWN: 0.0,
            Actions.LEFT: 0.0,
            Actions.RIGHT: 0.4,
        }
    )

    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    env.render_v(V, pi)
