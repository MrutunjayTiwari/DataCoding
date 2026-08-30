---
type: concept
status: active
tags: [numpy, arrays, vectorization]
updated: 2026-08-30
---

# NumPy

NumPy interview fluency is the ability to reason about shape, axis, dtype, and memory before reaching for loops.

## Mental model

An array is a typed block of values plus shape and strides. Most ML code is a sequence of shape-preserving elementwise operations, shape-changing transforms, and axis-reducing operations.

For `X.shape == (n, d)`:

- `X.mean(axis=0)` returns one value per feature: `(d,)`.
- `X.mean(axis=1)` returns one value per row: `(n,)`.
- `keepdims=True` retains a singleton axis, e.g. `(1, d)`, so later broadcasting is explicit.

## Core patterns

### Broadcasting

Align dimensions from the right. Dimensions are compatible when equal or when one equals `1`.

```python
mu = X.mean(axis=0, keepdims=True)  # (1, d)
X_centered = X - mu  # (n, d)
```

### Pairwise squared distances

For samples `X: (n, d)` and prototypes `C: (k, d)`:

```python
d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (n, k)
```

This single pattern powers [[wiki/algorithms/knn|KNN]] and [[wiki/algorithms/kmeans|K-means]].

### Ranking and aligned gather

Use `argsort` or `argpartition` to produce indices, then `take_along_axis` to gather aligned values. `argpartition` finds top-k candidates without fully sorting the rest.

### Views and copies

Basic slices commonly share memory; fancy/boolean indexing produces copies. `ravel` may share memory; `flatten` copies. Confirm with `np.shares_memory` when mutation matters.

### Numerical stability

- Stable softmax subtracts the row maximum before `exp`.
- Prefer `np.linalg.solve(A, b)` to `np.linalg.inv(A) @ b`.
- Add a small epsilon only when the mathematical denominator can genuinely approach zero; name why it is there.

## Failure modes

- A `(n,)` vector unexpectedly broadcasts across columns or rows.
- Reducing the wrong axis yields plausible but incorrect values.
- Repeated fancy indices do not accumulate with `out[idx] += values`; use `np.add.at` or an aggregation primitive.
- Integer dtype truncates a floating-point update.
- An unlabeled matrix output cannot be tied back to the producing expression.

## Interview drill

Implement stable softmax, top-k sorted indices, pairwise distances, precision/recall/F1, and one K-means update without Python loops over samples.

Executable reference: [NumPy interview refresher](../../notebooks/01-foundations/01_numpy_interview_refresher.ipynb).

## Connections

[[wiki/foundations/pandas|pandas]] · [[wiki/algorithms/from-scratch|From-scratch algorithms]] · [[wiki/workflows/pytorch-training-loop|PyTorch training loop]]

## Sources and provenance

Compiled from the legacy NumPy refresher variants and the advanced cells in the curated copy. The cleaned notebook consolidates those variants and labels every retained output.
