import numpy as np


class StandardScaler:
    def __init__(self):
        self.n_samples_seen = 0
        self.mean = None
        self.var = None
        self.std = None

    def fit(self, X):
        self.n_samples_seen = 0
        self.mean = None
        self.var = None
        self.std = None
        return self.partial_fit(X)

    def partial_fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        chunk_count = X.shape[0]
        if chunk_count == 0:
            return self

        chunk_mean = np.mean(X, axis=0)
        chunk_var = np.var(X, axis=0)

        if self.mean is None:
            self.n_samples_seen = chunk_count
            self.mean = chunk_mean
            self.var = chunk_var
            self.std = np.sqrt(self.var)
            return self

        total_count = self.n_samples_seen + chunk_count
        mean_diff = chunk_mean - self.mean
        old_sum_squares = self.var * self.n_samples_seen
        chunk_sum_squares = chunk_var * chunk_count
        new_sum_squares = (
            old_sum_squares
            + chunk_sum_squares
            + mean_diff**2 * self.n_samples_seen * chunk_count / total_count
        )

        self.mean = self.mean + mean_diff * chunk_count / total_count
        self.var = new_sum_squares / total_count
        self.std = np.sqrt(self.var)
        self.n_samples_seen = total_count
        return self

    def transform(self, X):
        if self.mean is None:
            raise ValueError("StandardScaler must be fitted before transform().")
        X = np.asarray(X, dtype=float)
        # this avoid division by zero for constant columns
        return (X - self.mean) / (self.std + 1e-10)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # scale features to 0, 1
        return (X - self.min) / (self.max - self.min + 1e-10)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    def fit(self, X):
        X = np.asarray(X)
        self.categories = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X)
        encoded_cols = []

        for i, cats in enumerate(self.categories):
            col = X[:, i][:, None]
            encoded_cols.append((col == cats).astype(int))

        return np.hstack(encoded_cols)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SimpleImputer:
    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def fit(self, X):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # replaced NaNs with a constant value
        X[np.isnan(X)] = self.fill_value
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
