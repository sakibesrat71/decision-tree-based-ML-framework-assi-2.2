import numpy as np


class DecisionTreeClassifier:
    """A small depth-limited decision tree classifier."""

    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini", max_features=None, random_state=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.classes = None
        self.feature_medians = None
        self._X_seen = None
        self._y_seen = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X, y = self._validate_xy(X, y)
        self.classes = np.unique(y)
        self.feature_medians = self._column_medians(X)
        X = self._fill_missing(X)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk, y_chunk = self._validate_xy(X_chunk, y_chunk)

        if self._X_seen is None:
            self._X_seen = X_chunk.copy()
            self._y_seen = y_chunk.copy()
        else:
            self._X_seen = np.vstack([self._X_seen, X_chunk])
            self._y_seen = np.concatenate([self._y_seen, y_chunk])

        return self.fit(self._X_seen, self._y_seen)

    def predict(self, X):
        if self.root is None:
            raise ValueError("DecisionTreeClassifier must be fitted before predict().")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._fill_missing(X)
        return np.array([self._predict_one(row, self.root) for row in X])

    def _validate_xy(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        return X, y

    def _column_medians(self, X):
        medians = np.nanmedian(X, axis=0)
        return np.where(np.isnan(medians), 0.0, medians)

    def _fill_missing(self, X):
        X = np.asarray(X, dtype=float).copy()
        rows, cols = np.where(np.isnan(X))
        if rows.size:
            X[rows, cols] = self.feature_medians[cols]
        return X

    def _build_tree(self, X, y, depth):
        node = {"prediction": self._majority_class(y)}

        if depth >= self.max_depth or X.shape[0] < self.min_samples_split or np.unique(y).size == 1:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature, threshold, left_mask = split
        node.update(
            {
                "feature": feature,
                "threshold": threshold,
                "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
                "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
            }
        )
        return node

    def _best_split(self, X, y):
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_split = None

        for feature in self._candidate_features(X.shape[1]):
            values = np.unique(X[:, feature])
            if values.size <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                if not left_mask.any() or left_mask.all():
                    continue
                gain = self._information_gain(y, left_mask, parent_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold, left_mask)

        return best_split

    def _candidate_features(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, str):
            if self.max_features != "sqrt":
                raise ValueError("max_features must be None, 'sqrt', or an integer.")
            count = max(1, int(np.sqrt(n_features)))
        else:
            count = min(n_features, max(1, int(self.max_features)))
        return self.rng.choice(n_features, size=count, replace=False)

    def _information_gain(self, y, left_mask, parent_impurity):
        left_y = y[left_mask]
        right_y = y[~left_mask]
        left_weight = left_y.size / y.size
        right_weight = right_y.size / y.size
        child_impurity = left_weight * self._impurity(left_y) + right_weight * self._impurity(right_y)
        return parent_impurity - child_impurity

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities**2)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict_one(self, row, node):
        if "feature" not in node:
            return node["prediction"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])
        return self._predict_one(row, node["right"])
