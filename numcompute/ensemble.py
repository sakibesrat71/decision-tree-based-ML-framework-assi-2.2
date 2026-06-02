import numpy as np

from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """A bagging-style ensemble built from decision trees."""

    def __init__(
        self,
        n_estimators=5,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees = []
        self.classes = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        self.classes = np.unique(y)
        self.trees = self._make_trees()

        for tree in self.trees:
            X_sample, y_sample = self._sample_rows(X, y)
            tree.fit(X_sample, y_sample)

        return self

    def predict(self, X):
        if not self.trees:
            raise ValueError("EnsembleClassifier must be fitted before predict().")

        predictions = np.vstack([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(self._majority_vote, axis=0, arr=predictions)

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk).ravel()
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)
        if X_chunk.shape[0] != y_chunk.shape[0]:
            raise ValueError("X_chunk and y_chunk must contain the same number of rows.")

        if not self.trees:
            self.classes = np.unique(y_chunk)
            self.trees = self._make_trees()
        else:
            self.classes = np.unique(np.concatenate([self.classes, np.unique(y_chunk)]))

        for tree in self.trees:
            X_sample, y_sample = self._sample_rows(X_chunk, y_chunk)
            tree.partial_fit(X_sample, y_sample)

        return self

    def _make_trees(self):
        seeds = self.rng.integers(0, 1_000_000, size=self.n_estimators)
        trees = []
        for seed in seeds:
            trees.append(
                DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    criterion=self.criterion,
                    max_features=self.max_features,
                    random_state=int(seed),
                )
            )
        return trees

    def _sample_rows(self, X, y):
        if not self.bootstrap:
            return X, y
        indices = self.rng.integers(0, X.shape[0], size=X.shape[0])
        return X[indices], y[indices]

    def _majority_vote(self, predictions):
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]
