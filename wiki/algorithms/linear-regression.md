---
type: concept
status: active
tags: [linear-regression, regression, gradient-descent]
updated: 2026-08-30
---

# Linear Regression

Linear regression predicts a continuous target with an affine function.

## Mental model

For `X: (n, d)`, weights `w: (d,)`, and intercept `b`, prediction is `X @ w + b`. Ordinary least squares minimizes:

$$\frac{1}{n}\sum_i (\hat y_i-y_i)^2$$

The batch gradient with respect to `w` is `2 * X.T @ residual / n`; the intercept gradient is the mean residual times two.

## Core patterns

- Closed form: solve $(X^TX + \lambda I)w = X^Ty$; prefer `solve` over explicit matrix inversion.
- Gradient descent: initialize weights, compute residual, compute gradient, update, and monitor loss.
- Standardize features when a shared learning rate must work across different scales.
- Do not regularize the intercept unless that choice is explicit.
- Evaluate with held-out MAE/RMSE/R²; training error only checks optimization.

## Failure modes

- Rank-deficient design matrix breaks an unregularized normal equation.
- A high learning rate diverges; a low rate appears frozen.
- Features on different scales make convergence uneven.
- Evaluating on training data disguises poor generalization.
- R² alone can hide practically large errors.

## Interview drill

Implement both closed form and gradient descent, recover known synthetic coefficients, then add L2 regularization without penalizing the bias.

Implementation: [`LinearRegressionGD`](../../src/datacoding/algorithms/linear_models.py). Notebook: [Linear models](../../notebooks/02-algorithms-from-scratch/01_linear_models.ipynb).

## Connections

[[wiki/algorithms/logistic-regression|Logistic regression]] · [[wiki/workflows/tabular-ml|Tabular ML]] · [[wiki/algorithms/pca|PCA]]
