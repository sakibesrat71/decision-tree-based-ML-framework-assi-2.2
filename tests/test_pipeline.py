import unittest

import numpy as np

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler


class DummyClassifier:
    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.label = values[np.argmax(counts)]
        return self

    def partial_fit(self, X, y):
        return self.fit(X, y)

    def predict(self, X):
        return np.full(len(X), self.label)


class TestPipeline(unittest.TestCase):
    def test_pipeline_fit_returns_self(self):
        pipe = Pipeline([("scale", StandardScaler()), ("model", DummyClassifier())])

        result = pipe.fit([[1, 2], [3, 4]], [0, 1])

        self.assertIs(result, pipe)

    def test_pipeline_predict_uses_final_model(self):
        pipe = Pipeline([("scale", StandardScaler()), ("model", DummyClassifier())])

        pipe.fit([[1, 2], [3, 4], [5, 6]], [1, 1, 0])

        np.testing.assert_array_equal(pipe.predict([[7, 8], [9, 10]]), [1, 1])

    def test_pipeline_partial_fit_updates_transformers_and_model(self):
        pipe = Pipeline([("scale", StandardScaler()), ("model", DummyClassifier())])

        pipe.partial_fit([[1, 2], [3, 4]], [0, 0])
        pipe.partial_fit([[5, 6], [7, 8]], [1, 1])

        self.assertEqual(pipe.steps[0][1].n_samples_seen, 4)
        np.testing.assert_array_equal(pipe.predict([[9, 10]]), [1])

    def test_pipeline_requires_steps(self):
        with self.assertRaises(ValueError):
            Pipeline([])


if __name__ == "__main__":
    unittest.main()
