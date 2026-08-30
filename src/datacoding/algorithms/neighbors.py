"""K-nearest-neighbors classifier from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from datacoding.algorithms._validation import as_1d, as_2d_float, require_fitted


@dataclass
class KNNClassifier:
    """Euclidean KNN with deterministic majority-vote tie breaking."""

    n_neighbors: int = 5

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNNClassifier:
        X_array = as_2d_float(X)
        if self.n_neighbors < 1 or self.n_neighbors > len(X_array):
            raise ValueError("n_neighbors must be between 1 and the number of training rows")
        self.X_train_ = X_array.copy()
        self.y_train_ = as_1d(y, n_samples=len(X_array)).copy()
        self.classes_ = np.unique(self.y_train_)
        self.n_features_in_ = X_array.shape[1]
        return self

    def _neighbor_indices(self, X: ArrayLike) -> NDArray[np.int64]:
        require_fitted(self, "X_train_")
        X_array = as_2d_float(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count does not match training data")
        squared_distances = np.sum(
            (X_array[:, None, :] - self.X_train_[None, :, :]) ** 2,
            axis=2,
        )
        candidates = np.argpartition(
            squared_distances,
            kth=self.n_neighbors - 1,
            axis=1,
        )[:, : self.n_neighbors]
        candidate_distances = np.take_along_axis(squared_distances, candidates, axis=1)
        order = np.argsort(candidate_distances, axis=1, kind="stable")
        return np.take_along_axis(candidates, order, axis=1)

    def predict(self, X: ArrayLike) -> NDArray:
        indices = self._neighbor_indices(X)
        votes = self.y_train_[indices]
        predictions = []
        for row in votes:
            counts = np.array([np.sum(row == label) for label in self.classes_])
            predictions.append(self.classes_[np.argmax(counts)])
        return np.asarray(predictions)
