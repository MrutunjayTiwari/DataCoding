---
type: concept
status: active
tags: [perceptron, classification, online-learning]
updated: 2026-08-30
---

# Perceptron

The perceptron is an online linear classifier that updates only when a sample is misclassified.

## Mental model

With labels `y ∈ {-1, +1}`, a sample is correct when `y * (x @ w + b) > 0`. Otherwise:

```python
w = w + learning_rate * y * x
b = b + learning_rate * y
```

A positive mistake pulls the boundary toward the sample; a negative mistake pushes it away.

## Core patterns

- Shuffle order may change the learned separator; seed it if shuffling.
- Stop early when an epoch contains zero mistakes.
- Track mistakes per epoch as the primary training diagnostic.
- State that convergence is guaranteed only for linearly separable data under the classic assumptions.

## Failure modes

- Using `{0, 1}` labels makes the update for class zero vanish.
- Non-separable data cycles without a maximum-epoch guard.
- A zero score is not assigned consistently.
- Training order is hidden, making runs irreproducible.

## Interview drill

Trace two updates by hand, implement the estimator, and explain how logistic regression differs: probability/loss optimization versus mistake-triggered updates.

Implementation: [`PerceptronClassifier`](../../src/datacoding/algorithms/linear_models.py). Notebook: [Linear models](../../notebooks/02-algorithms-from-scratch/01_linear_models.ipynb).

## Connections

[[wiki/algorithms/logistic-regression|Logistic regression]] · [[wiki/workflows/pytorch-training-loop|Neural training loops]] · [[wiki/foundations/python-oop|Estimator state]]
