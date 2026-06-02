from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numcompute.ensemble import EnsembleClassifier
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import SimpleImputer, StandardScaler
from numcompute.stream import StreamTrainer
from numcompute.tree import DecisionTreeClassifier
from numcompute.visualise import compare_models, plot_metric_over_time


def make_stream(n_samples=160, random_state=11):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 2))
    y = ((X[:, 0] * 0.7 + X[:, 1] * 0.4) > 0).astype(int)
    X[::19, 1] = np.nan
    return X, y


def run_stream(model, X, y, chunk_size=20):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    trainer = StreamTrainer(pipe)

    for start in range(0, len(X), chunk_size):
        X_chunk = X[start : start + chunk_size]
        y_chunk = y[start : start + chunk_size]
        trainer.fit_score_chunk(X_chunk, y_chunk)

    return trainer


def main():
    Path("outputs").mkdir(exist_ok=True)
    X, y = make_stream()

    tree = run_stream(DecisionTreeClassifier(max_depth=4, random_state=1), X, y)
    ensemble = run_stream(EnsembleClassifier(n_estimators=5, max_depth=4, random_state=2), X, y)

    print("Tree final cumulative accuracy:", round(tree.logs[-1]["cumulative_accuracy"], 3))
    print("Ensemble final cumulative accuracy:", round(ensemble.logs[-1]["cumulative_accuracy"], 3))

    plot_metric_over_time(
        tree.metric_history(),
        title="Decision tree chunk accuracy",
        ylabel="Accuracy",
        save_path="outputs/tree_accuracy.png",
        show=False,
    )
    compare_models(
        tree.metric_history(),
        ensemble.metric_history(),
        labels=("Tree", "Ensemble"),
        save_path="outputs/model_comparison.png",
        show=False,
    )


if __name__ == "__main__":
    main()
