# Report Outline

## Design decisions

Explain the chunk-based API, the pipeline update flow, the decision tree split
logic, and why the ensemble uses bagging.

## Testing and edge cases

Mention the tests for preprocessing, statistics, metrics, pipeline, tree,
ensemble, stream trainer, and visualisation. Include the edge cases for NaNs,
shape mismatches, zero-variance scaling, rolling windows, and unseen categories.

## Benchmark results

Run:

```bash
python benchmark/stream_benchmark.py
```

Record the runtime and cumulative accuracy for the tree and ensemble.

## Reflection

Discuss the main trade-off: the tree rebuilds from accumulated chunks during
`partial_fit()`, which is easy to inspect and test, but less efficient than a
specialised online tree algorithm.
