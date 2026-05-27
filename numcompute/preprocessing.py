import numpy as np


class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
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