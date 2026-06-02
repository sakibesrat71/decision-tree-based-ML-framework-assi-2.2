import unittest

import numpy as np

from numcompute.ensemble import EnsembleClassifier
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.stream import StreamTrainer
from numcompute.tree import DecisionTreeClassifier


class TestStreamTrainer(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=float)
        self.y = np.array([0, 0, 1, 1, 1, 1])

    def test_fit_score_chunk_adds_log_entry(self):
        pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(max_depth=3))])
        trainer = StreamTrainer(pipe)

        log = trainer.fit_score_chunk(self.X, self.y)

        self.assertEqual(len(trainer.logs), 1)
        self.assertIn("chunk_accuracy", log)
        self.assertIn("memory_bytes", log)

    def test_metric_history_returns_requested_metric(self):
        pipe = Pipeline([("model", DecisionTreeClassifier(max_depth=3))])
        trainer = StreamTrainer(pipe)

        trainer.fit_score_chunk(self.X[:3], self.y[:3])
        trainer.fit_score_chunk(self.X[3:], self.y[3:])

        self.assertEqual(len(trainer.metric_history("cumulative_accuracy")), 2)

    def test_stream_trainer_logs_auc_when_model_has_probabilities(self):
        pipe = Pipeline([("model", DecisionTreeClassifier(max_depth=3))])
        trainer = StreamTrainer(pipe)

        log = trainer.fit_score_chunk(self.X, self.y)

        self.assertIsNotNone(log["metrics"]["auc"])

    def test_stream_trainer_supports_ensemble_model(self):
        pipe = Pipeline([("model", EnsembleClassifier(n_estimators=3, max_depth=3, random_state=5))])
        trainer = StreamTrainer(pipe)

        log = trainer.fit_score_chunk(self.X, self.y)

        self.assertGreaterEqual(log["chunk_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
