import numpy as np

# import matplotlib.pyplot as plt


class Bandit:
    def __init__(self):
        self.__rate = np.random.rand()

    @property
    def rate(self) -> float:
        return self.__rate

    def play(self) -> int:
        if self.rate > np.random.rand():
            return 1
        return 0


if __name__ == "__main__":
    bandits = [Bandit() for _ in range(10)]
    Qs = np.zeros(10)
    ns = np.zeros(10)

    [print(bandits[i].rate) for i in range(10)]

    for n in range(10):
        action = np.random.randint(0, 10)
        reward = bandits[action].play()
        ns[action] += 1
        Qs[action] += (reward - Qs[action]) / ns[action]
        print(f"No: {action}, count: {ns[action]}, Q: {Qs[action]}")
