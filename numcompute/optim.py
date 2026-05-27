import numpy as np


def gradient(f, x, h=1e-5, method="central"):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_f = x.copy()
        x_b = x.copy()

        if method == "central":
            x_f[i] += h
            x_b[i] -= h
            grad[i] = (f(x_f) - f(x_b)) / (2 * h)

        elif method == "forward":
            x_f[i] += h
            grad[i] = (f(x_f) - f(x)) / h

        else:
            raise ValueError("method must be 'central' or 'forward'")

    return grad


def jacobian(f, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    y = f(x)

    J = np.zeros((len(y), len(x)))

    for i in range(len(x)):
        x_f = x.copy()
        x_b = x.copy()

        x_f[i] += h
        x_b[i] -= h

        J[:, i] = (f(x_f) - f(x_b)) / (2 * h)

    return J