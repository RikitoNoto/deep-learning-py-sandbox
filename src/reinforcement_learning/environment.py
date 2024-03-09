from collections import defaultdict
from enum import auto, Enum
from typing import Optional
import numpy as np
from reinforcement_learning.renderer import Renderer


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorld:
    def __init__(self):
        # self.action_space = [0, 1, 2, 3]
        # self.action_meaning = {
        #     0: "UP",
        #     1: "DOWN",
        #     2: "LEFT",
        #     3: "RIGHT",
        # }

        self.reward_map = np.array([[0, 0, 0, 1.0], [0, None, 0, -1.0], [0, 0, 0, 0]])
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return [action for action in Actions]

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action: Actions):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action.value]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state == self.goal_state

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(
        self,
        v=None,
        policy: Optional[defaultdict[tuple[int, int], dict[Actions, float]]] = None,
        print_value=True,
    ):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_state)

        converted_policy: Optional[dict[tuple[int, int], dict[int, float]]] = None
        if policy:
            converted_policy = self.__convert_policy_actions_to_index(policy)

        renderer.render_v(v, converted_policy, print_value)

    def __convert_policy_actions_to_index(
        self, policy: defaultdict[tuple[int, int], dict[Actions, float]]
    ) -> defaultdict[tuple[int, int], dict[int, float]]:
        converted_dict: dict[tuple[int, int], dict[int, float]] = defaultdict(
            lambda: dict(
                [(key.value, value) for key, value in policy.default_factory().items()]
            )
        )
        for state, pi in policy.items():
            converted_dict[state] = {}
            for action, probability in pi.items():
                converted_dict[state][action.value] = probability
        return converted_dict

    def render_q(self, q=None, print_value=True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_state)
        renderer.render_q(q, print_value)
