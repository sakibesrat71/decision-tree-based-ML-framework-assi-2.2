import numpy as np


def mean(x):
    x = np.asarray(x, dtype=float)
    return np.mean(x)


def variance(x):
    x = np.asarray(x, dtype=float)
    return np.var(x)


def standard_deviation(x):
    x = np.asarray(x, dtype=float)
    return np.std(x)


def median(x):
    x = np.asarray(x, dtype=float)
    return np.median(x)


def min_val(x):
    x = np.asarray(x, dtype=float)
    return np.min(x)


def max_val(x):
    x = np.asarray(x, dtype=float)
    return np.max(x)


def histogram(x, bins=10):
    x = np.asarray(x, dtype=float)
    return np.histogram(x, bins=bins)


def quantiles(x, q):
    x = np.asarray(x, dtype=float)
    return np.quantile(x, q)