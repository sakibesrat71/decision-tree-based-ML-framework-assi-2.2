import numpy as np


def rank(data, method="average"):
    x = np.asarray(data)
    n = len(x)

    order = np.argsort(x)
    ranks = np.zeros(n, dtype=float)

    sorted_x = x[order]

    if method == "ordinal":
        # assigns ranks based purely on sorted position (no tie handling)
        ranks[order] = np.arange(1, n + 1)
        return ranks

    i = 0
    current_rank = 1

    while i < n:
        j = i

        # finds range of equal values (handles ties)
        while j + 1 < n and sorted_x[j] == sorted_x[j + 1]:
            j += 1

        if method == "average":
            avg_rank = (i + j) / 2 + 1
            ranks[order[i:j + 1]] = avg_rank

        elif method == "dense":
            ranks[order[i:j + 1]] = current_rank
            current_rank += 1

        else:
            raise ValueError("method must be 'average', 'dense', or 'ordinal'")

        i = j + 1

    return ranks


def percentile(data, q, interpolation="linear"):
    x = np.asarray(data, dtype=float)

    return np.percentile(x, q, method=interpolation)