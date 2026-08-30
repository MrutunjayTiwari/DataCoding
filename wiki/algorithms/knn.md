---
type: concept
status: active
tags: [knn, classification, distance]
updated: 2026-08-30
---

# K-Nearest Neighbors

KNN predicts from the labels of the closest stored training samples.

## Mental model

KNN has almost no computational training step: `fit` validates and stores `X_train` and `y_train`. Prediction computes distances from `m` query rows to `n` training rows, selects `k`, and votes.

For `X_query: (m, d)` and `X_train: (n, d)`, broadcasted squared Euclidean distances have shape `(m, n)`.

## Core patterns

- Use squared distance when only ordering matters; the square root does not change neighbors.
- Use `argpartition` for candidate neighbors, then sort only those candidates when ordered distances are needed.
- Define deterministic tie-breaking, such as the smallest sorted class label.
- Standardize features using training statistics; distance is scale-sensitive.
- Choose `k` on validation data. Small `k` is flexible/noisy; large `k` is smooth/biased.

## Failure modes

- An ID-like or high-scale feature dominates distance.
- Data leakage occurs when scaling is fitted on all rows.
- `k` exceeds the number of training samples.
- A full `(m, n, d)` difference tensor exhausts memory for large data; use chunking or algebraic distance expansion.
- Tie behavior changes across runs or label ordering.

## Interview drill

Implement a 3-NN classifier, test a hand-computable example, explain fit/predict complexity, then add distance-weighted voting.

Implementation: [`KNNClassifier`](../../src/datacoding/algorithms/neighbors.py). Notebook: [KNN and K-means](../../notebooks/02-algorithms-from-scratch/02_knn_kmeans.ipynb).

## Connections

[[wiki/foundations/numpy|Broadcasting]] · [[wiki/algorithms/kmeans|K-means]] · [[wiki/workflows/sklearn-pipelines|Scaling in pipelines]]
