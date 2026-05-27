import numpy as np


def read_csv(filepath, delimiter=",", skip_header=True, fill_value=np.nan):
    # use numpy's loader for reliability and built-in missing value handling
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        skip_header=1 if skip_header else 0,
        dtype=float,
        filling_values=fill_value,
        autostrip=True
    )
    return data


def fill_missing_mean(X):
    X = np.asarray(X, dtype=float)

    # compute column-wise means ignoring NaNs
    col_means = np.nanmean(X, axis=0)

    # replace NaNs with corresponding column means
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    return X