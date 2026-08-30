---
type: concept
status: active
tags: [logistic-regression, classification, gradient-descent]
updated: 2026-08-30
---

# Logistic Regression

Logistic regression models a binary probability by applying a sigmoid to a linear score.

## Mental model

$$p(y=1\mid x)=\sigma(x^Tw+b),\qquad \sigma(z)=\frac{1}{1+e^{-z}}$$

Binary cross-entropy penalizes confident wrong probabilities. With `X: (n, d)`, the batch weight gradient is `X.T @ (p - y) / n`.

## Core patterns

- Clip logits before `exp`, or use a stable sigmoid implementation.
- Clip probabilities only for the logarithm in a hand-written loss.
- `predict_proba` returns probabilities; `predict` applies a decision threshold.
- Tune thresholds on validation data using the real cost of false positives/negatives.
- Standardize continuous inputs for reliable gradient descent.

## Failure modes

- Accuracy is misleading for severe imbalance.
- A hard 0.5 threshold is treated as part of the learned model rather than a decision policy.
- Labels use `{-1, +1}` while the loss assumes `{0, 1}`.
- Regularization accidentally penalizes the intercept.
- Perfect separation pushes coefficient magnitude upward without improving classifications.

## Interview drill

Implement `fit`, `predict_proba`, and `predict`; verify probabilities sum to one and compare accuracy with log loss on held-out data.

Implementation: [`LogisticRegressionGD`](../../src/datacoding/algorithms/linear_models.py). Notebook: [Linear models](../../notebooks/02-algorithms-from-scratch/01_linear_models.ipynb).

## Connections

[[wiki/algorithms/linear-regression|Linear regression]] · [[wiki/algorithms/perceptron|Perceptron]] · [[wiki/workflows/tabular-ml|Classification evaluation]]
