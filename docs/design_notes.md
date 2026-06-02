# Design Notes

## Streaming API

Most components support `partial_fit()` so a caller can pass new chunks as they
arrive. The pipeline updates each transformer, transforms the chunk, and then
updates the final model.

## Decision tree

The decision tree is depth-limited and supports Gini or entropy impurity. For
`partial_fit()`, it stores the chunks seen so far and rebuilds a small tree after
each update. This keeps the implementation easy to understand and reliable for
assignment-sized streams.

## Ensemble

The ensemble uses bagging. Each tree receives a bootstrap sample, and prediction
uses majority vote. `max_features="sqrt"` gives random-forest style feature
selection at each split.

## Numerical and edge-case handling

- Scalers use an epsilon when dividing by standard deviation or feature range.
- The imputer can update running column means.
- The decision tree fills missing feature values with learned medians.
- Metrics handle empty state and shape mismatches.
- Rolling windows are supported in metrics and streaming stats.

## Trade-offs

Rebuilding the tree during streaming updates is simpler than a specialised
Hoeffding tree. It is slower on large streams, but it makes the algorithm clear
and produces stable behaviour for this framework.
