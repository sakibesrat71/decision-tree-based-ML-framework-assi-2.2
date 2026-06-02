import unittest

import matplotlib

matplotlib.use("Agg")

from numcompute.visualise import compare_models, plot_metric_over_time, plot_predictions_vs_ground_truth


class TestVisualise(unittest.TestCase):
    def test_plot_metric_over_time_returns_figure(self):
        figure = plot_metric_over_time([0.5, 0.75], show=False)

        self.assertIsNotNone(figure)

    def test_compare_models_returns_figure(self):
        figure = compare_models([0.5, 0.7], [0.6, 0.8], labels=("Tree", "Ensemble"), show=False)

        self.assertIsNotNone(figure)

    def test_predictions_plot_returns_figure(self):
        figure = plot_predictions_vs_ground_truth([0, 1], [0, 1], show=False)

        self.assertIsNotNone(figure)


if __name__ == "__main__":
    unittest.main()
