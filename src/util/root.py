def f(x: float, value) -> float:
    return x**2 - value


def diff(x: float) -> float:
    return 2 * x


def section(x: float, value) -> float:
    return f(x, value) - (diff(x) * x)


def newton(x: float, value) -> float:
    """
    ニュートン法
    """
    return -section(x, value) / diff(x)


target_value = 3  # 平方根を求めたい数値
x = 2
for i in range(100):
    x = newton(x, target_value)
    print(x)
