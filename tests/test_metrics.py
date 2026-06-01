import unittest

import numpy as np

from numcompute.metrics import StreamingClassificationMetrics


class TestStreamingMetrics(unittest.TestCase):
    def test_metrics_accumulate_across_chunks(self):
        metric = StreamingClassificationMetrics()

        metric.update([1, 0], [1, 1])
        metric.update([1], [1])
        result = metric.result()

        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        np.testing.assert_array_equal(result["confusion_matrix"], np.array([[0, 1], [0, 2]]))

    def test_reset_clears_seen_values(self):
        metric = StreamingClassificationMetrics()

        metric.update([1], [1])
        metric.reset()

        self.assertEqual(metric.result()["accuracy"], 0.0)

    def test_rolling_window_uses_latest_values(self):
        metric = StreamingClassificationMetrics(window_size=2)

        metric.update([1, 1], [1, 0])
        metric.update([0, 0], [0, 0])

        self.assertAlmostEqual(metric.result()["accuracy"], 1.0)

    def test_auc_is_reported_when_scores_are_available(self):
        metric = StreamingClassificationMetrics()

        metric.update([0, 1], [0, 1], [0.1, 0.9])

        self.assertIsNotNone(metric.result()["auc"])

    def test_update_rejects_mismatched_prediction_lengths(self):
        metric = StreamingClassificationMetrics()

        with self.assertRaises(ValueError):
            metric.update([0, 1], [0])

    def test_update_rejects_mismatched_score_lengths(self):
        metric = StreamingClassificationMetrics()

        with self.assertRaises(ValueError):
            metric.update([0, 1], [0, 1], [0.5])


if __name__ == "__main__":
    unittest.main()
