import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def normal(x, mu=0, sigma=1):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


x = np.random.random(100)  # np.arange(-2, 2, 0.01)
mu = np.arange(-2, 2, 0.01)


y = np.log(np.array([np.sum(normal(x, mu=m)) for m in mu]))


ax: Axes = plt.axes()
ax.plot(mu, y)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
