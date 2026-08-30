"""Principal component analysis via singular value decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from datacoding.algorithms._validation import as_2d_float, require_fitted


@dataclass
class PCA:
    """Small PCA estimator that mirrors the essential sklearn interface."""

    n_components: int = 2

    def fit(self, X: ArrayLike) -> PCA:
        X_array = as_2d_float(X)
        max_components = min(X_array.shape)
        if not 1 <= self.n_components <= max_components:
            raise ValueError(f"n_components must be in [1, {max_components}]")

        self.mean_ = X_array.mean(axis=0)
        centered = X_array - self.mean_
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        explained_variance = singular_values**2 / (len(X_array) - 1)
        self.components_ = right_vectors[: self.n_components]
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / explained_variance.sum()
        self.n_features_in_ = X_array.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        require_fitted(self, "components_")
        X_array = as_2d_float(X)
        return (X_array - self.mean_) @ self.components_.T

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: ArrayLike) -> NDArray[np.float64]:
        require_fitted(self, "components_")
        reduced = as_2d_float(X_reduced)
        if reduced.shape[1] != self.n_components:
            raise ValueError("Reduced feature count does not match n_components")
        return reduced @ self.components_ + self.mean_
