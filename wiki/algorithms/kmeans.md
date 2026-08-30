---
type: concept
status: active
tags: [kmeans, clustering, unsupervised]
updated: 2026-08-30
---

# K-Means

K-means alternates nearest-center assignment and centroid recomputation to reduce within-cluster squared distance.

## Mental model

1. Initialize `k` centers with shape `(k, d)`.
2. Compute sample-center squared distances `(n, k)`.
3. Assign each sample to its closest center.
4. Replace each center with the mean of assigned samples.
5. Stop when centers move less than a tolerance or the iteration limit is reached.

The objective (inertia) is the sum of squared distances from each point to its assigned center.

## Core patterns

- Seed initialization for reproducibility; multiple restarts reduce local-minimum risk.
- Standardize features when scale should not determine clusters.
- Handle empty clusters explicitly: re-seed them rather than divide by zero.
- Recompute labels after the final center update before reporting inertia.
- Cluster numbers have no semantic ordering and can permute between runs.

## Failure modes

- K-means assumes roughly compact, Euclidean clusters and performs poorly on curved or unequal-density structures.
- Outliers pull arithmetic means strongly.
- Comparing raw label IDs to ground truth is meaningless without label alignment.
- A too-small tolerance wastes iterations; a too-large one stops early.
- One initialization gives an unstable conclusion.

## Interview drill

Implement one assignment/update iteration, extend it to convergence, add empty-cluster handling, and state time complexity `O(iterations * n * k * d)`.

Implementation: [`KMeans`](../../src/datacoding/algorithms/clustering.py). Notebook: [KNN and K-means](../../notebooks/02-algorithms-from-scratch/02_knn_kmeans.ipynb).

## Connections

[[wiki/algorithms/knn|KNN]] · [[wiki/algorithms/pca|PCA]] · [[wiki/foundations/numpy|Pairwise distances]]
