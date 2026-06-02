import matplotlib.pyplot as plt
import numpy as np


def _finish_plot(save_path=None, show=True):
    figure = plt.gcf()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return figure


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric", save_path=None, show=True):
    values = np.asarray(metric_values, dtype=float)

    plt.figure()
    plt.plot(np.arange(1, values.size + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    return _finish_plot(save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), save_path=None, show=True):
    values1 = np.asarray(metric1, dtype=float)
    values2 = np.asarray(metric2, dtype=float)

    plt.figure()
    plt.plot(np.arange(1, values1.size + 1), values1, marker="o", label=labels[0])
    plt.plot(np.arange(1, values2.size + 1), values2, marker="s", label=labels[1])
    plt.title("Streaming model comparison")
    plt.xlabel("Chunk")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return _finish_plot(save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, save_path=None, show=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positions = np.arange(y_true.size)

    plt.figure()
    plt.scatter(positions, y_true, label="Ground truth", alpha=0.8)
    plt.scatter(positions, y_pred, label="Prediction", marker="x", alpha=0.8)
    plt.title("Predictions vs ground truth")
    plt.xlabel("Sample")
    plt.ylabel("Class")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return _finish_plot(save_path, show)
