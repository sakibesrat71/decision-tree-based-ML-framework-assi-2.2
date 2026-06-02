from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numcompute.ensemble import EnsembleClassifier
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.stream import StreamTrainer
from numcompute.tree import DecisionTreeClassifier


def make_dataset(n_samples=500, n_features=4, random_state=21):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    y = (X @ weights > 0).astype(int)
    return X, y


def time_stream(model, X, y, chunk_size=50):
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    trainer = StreamTrainer(pipe)

    start_time = time.perf_counter()
    for start in range(0, len(X), chunk_size):
        X_chunk = X[start : start + chunk_size]
        y_chunk = y[start : start + chunk_size]
        trainer.fit_score_chunk(X_chunk, y_chunk)
    elapsed = time.perf_counter() - start_time

    return elapsed, trainer.logs[-1]["cumulative_accuracy"]


def main():
    X, y = make_dataset()
    tree_time, tree_accuracy = time_stream(DecisionTreeClassifier(max_depth=5, random_state=1), X, y)
    ensemble_time, ensemble_accuracy = time_stream(
        EnsembleClassifier(n_estimators=7, max_depth=5, random_state=2),
        X,
        y,
    )

    print("model,time_seconds,cumulative_accuracy")
    print(f"tree,{tree_time:.6f},{tree_accuracy:.4f}")
    print(f"ensemble,{ensemble_time:.6f},{ensemble_accuracy:.4f}")


if __name__ == "__main__":
    main()
