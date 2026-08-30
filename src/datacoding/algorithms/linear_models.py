"""Linear regression, logistic regression, and perceptron from scratch."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from datacoding.algorithms._validation import as_1d, as_2d_float, require_fitted


def _with_bias(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack([X, np.ones(X.shape[0])])


@dataclass
class LinearRegressionGD:
    """Batch-gradient-descent linear regression with optional L2 penalty."""

    learning_rate: float = 0.05
    max_iter: int = 2_000
    l2: float = 0.0
    tolerance: float = 1e-10
    loss_history_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegressionGD:
        X_array = as_2d_float(X)
        y_array = as_1d(y, n_samples=len(X_array)).astype(float)
        X_bias = _with_bias(X_array)
        weights = np.zeros(X_bias.shape[1], dtype=float)
        self.loss_history_ = []

        for _ in range(self.max_iter):
            residual = X_bias @ weights - y_array
            penalty = weights.copy()
            penalty[-1] = 0.0  # do not regularize the intercept
            gradient = 2.0 * (X_bias.T @ residual) / len(X_bias) + 2.0 * self.l2 * penalty
            next_weights = weights - self.learning_rate * gradient
            loss = float(np.mean(residual**2) + self.l2 * np.sum(penalty**2))
            self.loss_history_.append(loss)
            if np.linalg.norm(next_weights - weights) <= self.tolerance:
                weights = next_weights
                break
            weights = next_weights

        self.coef_ = weights[:-1]
        self.intercept_ = float(weights[-1])
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        require_fitted(self, "coef_")
        X_array = as_2d_float(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count does not match training data")
        return X_array @ self.coef_ + self.intercept_


@dataclass
class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent."""

    learning_rate: float = 0.1
    max_iter: int = 2_000
    l2: float = 0.0
    tolerance: float = 1e-10
    loss_history_: list[float] = field(default_factory=list, init=False)

    @staticmethod
    def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegressionGD:
        X_array = as_2d_float(X)
        y_array = as_1d(y, n_samples=len(X_array)).astype(float)
        if not set(np.unique(y_array)).issubset({0.0, 1.0}):
            raise ValueError("LogisticRegressionGD expects binary labels encoded as 0 and 1")

        X_bias = _with_bias(X_array)
        weights = np.zeros(X_bias.shape[1], dtype=float)
        self.loss_history_ = []

        for _ in range(self.max_iter):
            probability = self._sigmoid(X_bias @ weights)
            penalty = weights.copy()
            penalty[-1] = 0.0
            gradient = X_bias.T @ (probability - y_array) / len(X_bias) + 2.0 * self.l2 * penalty
            next_weights = weights - self.learning_rate * gradient
            clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
            loss = -np.mean(y_array * np.log(clipped) + (1 - y_array) * np.log(1 - clipped))
            self.loss_history_.append(float(loss + self.l2 * np.sum(penalty**2)))
            if np.linalg.norm(next_weights - weights) <= self.tolerance:
                weights = next_weights
                break
            weights = next_weights

        self.coef_ = weights[:-1]
        self.intercept_ = float(weights[-1])
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        require_fitted(self, "coef_")
        X_array = as_2d_float(X)
        positive = self._sigmoid(X_array @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> NDArray[np.int64]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


@dataclass
class PerceptronClassifier:
    """Classic online perceptron for labels encoded as -1 and +1."""

    learning_rate: float = 1.0
    max_epochs: int = 100

    def fit(self, X: ArrayLike, y: ArrayLike) -> PerceptronClassifier:
        X_array = as_2d_float(X)
        y_array = as_1d(y, n_samples=len(X_array)).astype(int)
        if set(np.unique(y_array)) != {-1, 1}:
            raise ValueError("PerceptronClassifier expects both -1 and +1 labels")

        X_bias = _with_bias(X_array)
        weights = np.zeros(X_bias.shape[1], dtype=float)
        self.mistakes_per_epoch_ = []

        for _ in range(self.max_epochs):
            mistakes = 0
            for row, target in zip(X_bias, y_array, strict=True):
                if target * float(row @ weights) <= 0.0:
                    weights += self.learning_rate * target * row
                    mistakes += 1
            self.mistakes_per_epoch_.append(mistakes)
            if mistakes == 0:
                break

        self.coef_ = weights[:-1]
        self.intercept_ = float(weights[-1])
        self.n_features_in_ = X_array.shape[1]
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        require_fitted(self, "coef_")
        X_array = as_2d_float(X)
        return X_array @ self.coef_ + self.intercept_

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
