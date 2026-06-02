import numpy as np

from .metrics import StreamingClassificationMetrics


class StreamTrainer:
    """Manage chunk-wise training, scoring, and logging."""

    def __init__(self, pipeline, metric=None):
        self.pipeline = pipeline
        self.metric = metric if metric is not None else StreamingClassificationMetrics()
        self.logs = []
        self.total_correct = 0
        self.total_seen = 0

    def fit_chunk(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.pipeline.partial_fit(X, y)
        return self

    def score_chunk(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        y_pred = self.pipeline.predict(X)
        y_score = self._positive_class_scores(X)
        self.metric.update(y, y_pred, y_score)

        correct = int(np.sum(y == y_pred))
        self.total_correct += correct
        self.total_seen += y.size

        entry = {
            "chunk": len(self.logs),
            "chunk_size": int(y.size),
            "chunk_accuracy": correct / y.size if y.size else 0.0,
            "cumulative_accuracy": self.total_correct / self.total_seen if self.total_seen else 0.0,
            "memory_bytes": self._memory_bytes(X, y, y_pred),
            "metrics": self.metric.result(),
        }
        self.logs.append(entry)
        return entry

    def fit_score_chunk(self, X, y):
        self.fit_chunk(X, y)
        return self.score_chunk(X, y)

    def metric_history(self, key="chunk_accuracy"):
        return [entry[key] for entry in self.logs]

    def _memory_bytes(self, *arrays):
        total = 0
        for array in arrays:
            total += np.asarray(array).nbytes
        return int(total)

    def _positive_class_scores(self, X):
        if not hasattr(self.pipeline, "steps"):
            return None

        model = self.pipeline.steps[-1][1]
        if not hasattr(model, "predict_proba"):
            return None

        data = self.pipeline._transform_steps(X)
        probabilities = model.predict_proba(data)
        if probabilities.shape[1] < 2:
            return None
        return probabilities[:, 1]
