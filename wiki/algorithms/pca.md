---
type: concept
status: active
tags: [pca, svd, dimensionality-reduction]
updated: 2026-08-30
---

# PCA via SVD

PCA rotates centered data onto orthogonal directions that capture decreasing variance.

## Mental model

For centered `X: (n, d)`, compute `U, S, Vt = svd(X, full_matrices=False)`. Rows of `Vt` are principal directions. The first `r` components form `(r, d)` and transform data as `X_centered @ components.T`.

Explained variance is `S**2 / (n - 1)`; the ratio divides selected variances by total variance.

## Core patterns

- Fit and store the training mean; apply the same mean to validation/test data.
- Standardize first when feature units should have comparable influence.
- Use SVD directly rather than forming a covariance matrix when numerical stability matters.
- `inverse_transform` helps interpret information loss.
- Component signs may flip with no change in the represented subspace.

## Failure modes

- Forgetting centering makes the origin drive components.
- Fitting PCA before a split leaks test-distribution information.
- Interpreting sign as intrinsically meaningful.
- Assuming high variance means high predictive value.
- Comparing components from differently scaled inputs.

## Interview drill

Implement `fit`, `transform`, and `inverse_transform`; verify transformed shape, explained-variance ratio, and reconstruction error.

Implementation: [`PCA`](../../src/datacoding/algorithms/decomposition.py). The NumPy refresher includes the SVD mechanics.

## Connections

[[wiki/foundations/numpy|NumPy linear algebra]] · [[wiki/algorithms/kmeans|K-means]] · [[wiki/workflows/sklearn-pipelines|Pipeline-safe transforms]]
