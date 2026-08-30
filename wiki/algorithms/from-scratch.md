---
type: map
status: active
tags: [algorithms, from-scratch, map]
updated: 2026-08-30
---

# From-Scratch Algorithm Map

Implementing algorithms from scratch is shape reasoning plus a small, testable state machine.

## Shared ritual

1. Define `X: (n, d)`, `y: (n,)`, prediction shape, and label encoding.
2. State the objective or decision rule.
3. Write the simplest correct baseline.
4. Separate `fit` from `predict` and store learned state with trailing underscores.
5. Test a tiny hand-computable case.
6. Test a seeded synthetic case.
7. Discuss complexity and one numerical or statistical failure mode.

## Families

### Optimization-based

- [[wiki/algorithms/linear-regression|Linear regression]]: minimize mean squared error.
- [[wiki/algorithms/logistic-regression|Logistic regression]]: minimize binary cross-entropy.
- [[wiki/algorithms/perceptron|Perceptron]]: update only on mistakes.

### Distance-based

- [[wiki/algorithms/knn|KNN]]: retain training examples and vote among neighbors.
- [[wiki/algorithms/kmeans|K-means]]: alternate assignment and centroid update.

### Decomposition

- [[wiki/algorithms/pca|PCA]]: center data and retain leading singular directions.

## Complexity snapshot

| Algorithm | Fit | Predict/transform | Important memory |
|---|---:|---:|---:|
| Linear/logistic | `O(iterations * n * d)` | `O(m * d)` | coefficients |
| Perceptron | `O(epochs * n * d)` | `O(m * d)` | coefficients |
| KNN | `O(1)` storage | `O(m * n * d)` | all training data |
| K-means | `O(iterations * n * k * d)` | `O(m * k * d)` | centers |
| PCA (full SVD) | shape-dependent, roughly cubic in smaller dimension | `O(m * d * components)` | mean and components |

Executable references: [linear models](../../notebooks/02-algorithms-from-scratch/01_linear_models.ipynb) and [KNN/K-means](../../notebooks/02-algorithms-from-scratch/02_knn_kmeans.ipynb).

## Connections

[[wiki/foundations/numpy|NumPy]] · [[wiki/foundations/python-oop|Python OOP]] · [[wiki/workflows/sklearn-pipelines|sklearn pipelines]]
