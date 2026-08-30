"""Shared input validation for the teaching estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class NotFittedError(RuntimeError):
    """Raised when prediction is attempted before fitting."""


def as_2d_float(X: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"X must be 2D; received shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature")
    if not np.isfinite(array).all():
        raise ValueError("X contains NaN or infinite values")
    return array


def as_1d(y: ArrayLike, *, n_samples: int) -> NDArray:
    array = np.asarray(y).reshape(-1)
    if len(array) != n_samples:
        raise ValueError(f"X has {n_samples} rows but y has {len(array)} values")
    return array


def require_fitted(instance: object, attribute: str) -> None:
    if not hasattr(instance, attribute):
        raise NotFittedError(f"{type(instance).__name__} must be fitted before prediction")
