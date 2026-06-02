import unittest

import numpy as np

from numcompute.ensemble import EnsembleClassifier


class TestEnsembleClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=float)
        self.y = np.array([0, 0, 1, 1, 1, 1])

    def test_fit_predict(self):
        model = EnsembleClassifier(n_estimators=5, max_depth=3, random_state=1)

        model.fit(self.X, self.y)

        self.assertEqual(model.predict([[2, 2]])[0], 1)

    def test_partial_fit_updates_trees(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=2)

        model.partial_fit(self.X[:3], self.y[:3])
        model.partial_fit(self.X[3:], self.y[3:])

        self.assertEqual(len(model.trees), 3)
        self.assertEqual(model.predict([[1, 1]])[0], 1)

    def test_predict_proba_shape(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=3)

        model.fit(self.X, self.y)

        self.assertEqual(model.predict_proba(self.X[:2]).shape, (2, 2))

    def test_can_disable_bootstrap(self):
        model = EnsembleClassifier(n_estimators=2, bootstrap=False, random_state=4)

        model.fit(self.X, self.y)

        self.assertEqual(len(model.trees), 2)

    def test_rejects_bad_shapes(self):
        model = EnsembleClassifier()

        with self.assertRaises(ValueError):
            model.fit([[1], [2]], [1])


if __name__ == "__main__":
    unittest.main()
