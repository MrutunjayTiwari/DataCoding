"""K-means clustering from scratch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from datacoding.algorithms._validation import as_2d_float, require_fitted


@dataclass
class KMeans:
    """Lloyd's K-means algorithm with deterministic seeded initialization."""

    n_clusters: int = 3
    max_iter: int = 100
    tolerance: float = 1e-4
    random_state: int = 42

    @staticmethod
    def _squared_distances(
        X: NDArray[np.float64], centers: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _initialize_centers(
        self, X: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Choose deterministic seeded K-means++ centers."""

        first_index = int(rng.integers(0, len(X)))
        centers = [X[first_index].copy()]

        while len(centers) < self.n_clusters:
            chosen = np.vstack(centers)
            closest_squared = self._squared_distances(X, chosen).min(axis=1)
            total = float(closest_squared.sum())
            if total == 0.0:
                # All rows coincide with an existing center; any row is equivalent.
                next_index = int(rng.integers(0, len(X)))
            else:
                next_index = int(rng.choice(len(X), p=closest_squared / total))
            centers.append(X[next_index].copy())

        return np.vstack(centers)

    def fit(self, X: ArrayLike) -> KMeans:
        X_array = as_2d_float(X)
        if not 1 <= self.n_clusters <= len(X_array):
            raise ValueError("n_clusters must be between 1 and the number of rows")

        rng = np.random.default_rng(self.random_state)
        centers = self._initialize_centers(X_array, rng)

        for iteration in range(1, self.max_iter + 1):
            distances = self._squared_distances(X_array, centers)
            labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()

            for cluster in range(self.n_clusters):
                members = X_array[labels == cluster]
                if len(members):
                    new_centers[cluster] = members.mean(axis=0)
                else:
                    # Re-seed an empty cluster at the point farthest from its assigned center.
                    nearest_distance = distances[np.arange(len(X_array)), labels]
                    new_centers[cluster] = X_array[np.argmax(nearest_distance)]

            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift <= self.tolerance:
                break

        final_distances = self._squared_distances(X_array, centers)
        self.cluster_centers_ = centers
        self.labels_ = np.argmin(final_distances, axis=1)
        self.inertia_ = float(np.sum(final_distances[np.arange(len(X_array)), self.labels_]))
        self.n_iter_ = iteration
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        require_fitted(self, "cluster_centers_")
        X_array = as_2d_float(X)
        return np.argmin(self._squared_distances(X_array, self.cluster_centers_), axis=1)

    def fit_predict(self, X: ArrayLike) -> NDArray[np.int64]:
        return self.fit(X).labels_
