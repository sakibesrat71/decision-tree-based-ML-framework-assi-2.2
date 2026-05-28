import numpy as np


#Classification Metrics

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return tp / (tp + fp + 1e-10)


def recall(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp / (tp + fn + 1e-10)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-10)


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp],
                     [fn, tp]])


#Regression Metric

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean((y_true - y_pred) ** 2)


#I implemented ROC curve and AUC to measure how well the model distinguishes between classes

def roc_curve(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    thresholds = np.sort(np.unique(y_scores))[::-1]

    tpr = []
    fpr = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr.append(tp / (tp + fn + 1e-10))
        fpr.append(fp / (fp + tn + 1e-10))

    return np.array(fpr), np.array(tpr), thresholds


def auc(fpr, tpr):
    return np.trapz(tpr, fpr)


class StreamingClassificationMetrics:
    """Accumulate binary classification metrics over data chunks."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true_chunk = np.asarray(y_true_chunk).ravel()
        y_pred_chunk = np.asarray(y_pred_chunk).ravel()

        if y_true_chunk.shape[0] != y_pred_chunk.shape[0]:
            raise ValueError("y_true_chunk and y_pred_chunk must have the same length.")

        self.y_true.extend(y_true_chunk.tolist())
        self.y_pred.extend(y_pred_chunk.tolist())
        return self

    def result(self):
        if not self.y_true:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": np.zeros((2, 2), dtype=int),
            }

        y_true = np.asarray(self.y_true)
        y_pred = np.asarray(self.y_pred)

        return {
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred),
            "recall": recall(y_true, y_pred),
            "f1": f1(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }
