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


class StreamingStats:
    """Track running mean and variance from incoming chunks."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update_stats(self, X_chunk):
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X = X[~np.isnan(X).any(axis=1)]
        if X.shape[0] == 0:
            return self

        chunk_count = X.shape[0]
        chunk_mean = np.mean(X, axis=0)
        chunk_m2 = np.var(X, axis=0) * chunk_count

        if self.mean is None:
            self.count = chunk_count
            self.mean = chunk_mean
            self.m2 = chunk_m2
            return self

        total_count = self.count + chunk_count
        mean_diff = chunk_mean - self.mean
        self.m2 = self.m2 + chunk_m2 + mean_diff**2 * self.count * chunk_count / total_count
        self.mean = self.mean + mean_diff * chunk_count / total_count
        self.count = total_count
        return self

    def result(self):
        if self.count == 0:
            return {"count": 0, "mean": None, "variance": None}

        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.m2 / self.count,
        }


def update_stats(X_chunk, stats_obj=None):
    if stats_obj is None:
        stats_obj = StreamingStats()
    return stats_obj.update_stats(X_chunk)
