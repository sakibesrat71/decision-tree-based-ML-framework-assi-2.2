# NumCompute Streaming Decision Trees

This project extends the original `numcompute` package into a small
decision-tree based machine learning framework for streaming data.

The framework uses only plain Python, NumPy, and matplotlib.

## Main components

- `numcompute.tree.DecisionTreeClassifier`: depth-limited Gini or entropy tree
- `numcompute.ensemble.EnsembleClassifier`: bagging-style ensemble of trees
- `numcompute.preprocessing`: streaming scalers, imputer, and one-hot encoder
- `numcompute.metrics`: batch and streaming classification metrics
- `numcompute.stats`: chunk-based running statistics
- `numcompute.pipeline.Pipeline`: chained transformers and models
- `numcompute.stream.StreamTrainer`: chunk training, scoring, and logging
- `numcompute.visualise`: reusable matplotlib plotting helpers

## Run tests

```bash
python -m unittest discover -s tests
```

or:

```bash
python run_tests.py
```

## Run the demo

```bash
python demo/stream_demo.py
```

The demo trains a single tree and an ensemble over chunks and saves plots in
`outputs/`.

## Run the benchmark

```bash
python benchmark/stream_benchmark.py
```

The benchmark prints runtime and cumulative accuracy for a single tree and an
ensemble under the same streaming scenario.

## Small usage example

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.stream import StreamTrainer
from numcompute.tree import DecisionTreeClassifier

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=4)),
])

trainer = StreamTrainer(pipe)
trainer.fit_chunk(X_chunk, y_chunk)
log = trainer.score_chunk(X_chunk, y_chunk)
```
