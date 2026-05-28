import unittest

import numpy as np

from numcompute.stats import StreamingStats, update_stats


class TestStreamingStats(unittest.TestCase):
    def test_running_mean_and_variance_across_chunks(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        stats = StreamingStats()

        stats.update_stats(X[:1])
        stats.update_stats(X[1:])
        result = stats.result()

        self.assertEqual(result["count"], 3)
        np.testing.assert_allclose(result["mean"], np.mean(X, axis=0))
        np.testing.assert_allclose(result["variance"], np.var(X, axis=0))

    def test_empty_stats_result_is_safe(self):
        result = StreamingStats().result()

        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["variance"])

    def test_quantiles_use_seen_values(self):
        stats = StreamingStats()

        stats.update_stats([1, 2, 3, 4])

        self.assertAlmostEqual(stats.quantiles(0.5).item(), 2.5)

    def test_histogram_counts_seen_values(self):
        stats = StreamingStats(bins=2)

        stats.update_stats([1, 2, 3, 4])
        counts, _ = stats.histogram()

        self.assertEqual(int(np.sum(counts)), 4)

    def test_update_stats_helper_creates_object(self):
        stats = update_stats([1, 2, 3])

        self.assertEqual(stats.result()["count"], 3)

    def test_rolling_window_keeps_latest_rows(self):
        stats = StreamingStats(window_size=2)

        stats.update_stats([[1], [2], [3]])
        result = stats.result()

        self.assertEqual(result["count"], 2)
        np.testing.assert_allclose(result["mean"], [2.5])


if __name__ == "__main__":
    unittest.main()
