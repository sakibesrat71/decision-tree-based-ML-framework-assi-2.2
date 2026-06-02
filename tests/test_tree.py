import unittest

import numpy as np

from numcompute.metrics import accuracy
from numcompute.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=float)
        self.y = np.array([0, 0, 1, 1, 1, 1])

    def test_fit_predict(self):
        model = DecisionTreeClassifier(max_depth=3)

        model.fit(self.X, self.y)

        self.assertGreaterEqual(accuracy(self.y, model.predict(self.X)), 0.8)

    def test_entropy_criterion(self):
        model = DecisionTreeClassifier(max_depth=3, criterion="entropy")

        model.fit(self.X, self.y)

        self.assertEqual(model.predict([[0, 0]])[0], 0)

    def test_partial_fit_updates_model(self):
        model = DecisionTreeClassifier(max_depth=3)

        model.partial_fit(self.X[:3], self.y[:3])
        model.partial_fit(self.X[3:], self.y[3:])

        self.assertEqual(model.predict([[2, 2]])[0], 1)

    def test_predict_proba_shape(self):
        model = DecisionTreeClassifier(max_depth=3)

        model.fit(self.X, self.y)

        self.assertEqual(model.predict_proba(self.X[:2]).shape, (2, 2))

    def test_handles_nan_with_feature_median(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        model = DecisionTreeClassifier(max_depth=2)

        model.fit(X, self.y)

        self.assertEqual(model.predict([[np.nan, 0]])[0], 0)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit([[1], [2]], [1])


if __name__ == "__main__":
    unittest.main()
