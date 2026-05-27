import numpy as np


def topk(x, k, largest=True, return_indices=False):
    x = np.asarray(x)

    if not 1 <= k <= x.size:
        raise ValueError("k must be between 1 and array size")

    # get k smallest or largest elements using argpartition
    if largest:
        indices = np.argpartition(x, -k)[-k:]
        indices = indices[np.argsort(x[indices])[::-1]]
    else:
        indices = np.argpartition(x, k)[:k]
        indices = indices[np.argsort(x[indices])]

    if return_indices:
        return x[indices], indices

    return x[indices]


def binary_search(x, target):
    x = np.asarray(x)

    left, right = 0, len(x) - 1

    while left <= right:
        mid = (left + right) // 2

        if x[mid] == target:
            return mid, True
        elif x[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # returns insertion index + not found
    return left, False


def stable_sort(x):
    x = np.asarray(x)
    return np.sort(x, kind="stable")