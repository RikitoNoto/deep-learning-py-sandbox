from typing import Optional
import numpy as np

import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, rate: Optional[float] = None):
        self.__rate = rate or np.random.rand()

    @property
    def rate(self) -> float:
        return self.__rate

    def play(self) -> int:
        if self.rate > np.random.rand():
            return 1
        return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self, n: int, steps: int):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


if __name__ == "__main__":

    def run(steps: int, epsilon: float) -> tuple[int, list[int], list[float]]:
        # bandits = [Bandit(i / 10 + i / 100) for i in range(10)]
        bandits = [Bandit() for _ in range(10)]
        agent = Agent(epsilon, 10)
        total_reward = 0
        total_rewards = []
        rates = []

        for step in range(steps):
            action = agent.get_action(step, steps)
            reward = bandits[action].play()
            agent.update(action, reward)
            total_reward += reward

            total_rewards.append(total_reward)
            rates.append(total_reward / (step + 1))
        return total_reward, total_rewards, rates

    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates = np.zeros((runs, steps))
    for i in range(runs):
        total_reward, total_rewards, rates = run(steps, epsilon)
        all_rates[i] = rates

    avg_rates = np.average(all_rates, axis=0)
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(avg_rates)
    plt.show()
