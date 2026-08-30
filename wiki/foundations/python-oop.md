---
type: concept
status: active
tags: [python, oop, estimators, design]
updated: 2026-08-30
---

# Python OOP for ML

Use OOP to make learned state explicit and reusable, not to wrap every function in a class.

## Mental model

An estimator object separates:

- **configuration** chosen before training (`learning_rate`, `n_neighbors`);
- **learned state** created by `fit` (`coef_`, `cluster_centers_`);
- **behavior** operating on state (`predict`, `transform`).

The trailing underscore convention makes learned attributes visually distinct from constructor parameters.

## Small estimator contract

```python
model = Estimator(hyperparameter=value)
model.fit(X_train, y_train)  # validates input and returns self
prediction = model.predict(X_test)
```

Prefer composition: a preprocessing object and estimator can be combined without either inheriting from the other. Use inheritance only for a stable shared contract, not merely to reuse a few lines.

## Design checklist

- Keep `__init__` cheap; store configuration only.
- Validate dimensionality and labels at the boundary.
- Never silently fit during `predict`.
- Copy training data when the algorithm must retain it, as [[wiki/algorithms/knn|KNN]] does.
- Seed randomness through a constructor parameter.
- Return NumPy arrays with documented shapes.
- Put reusable code in modules and demonstrations in notebooks.
- Test outcomes and edge cases rather than private implementation details.

## Failure modes

- Class-level mutable lists are shared between instances.
- Constructor code downloads data or performs training.
- Fitted state is indistinguishable from hyperparameters.
- One giant class loads data, trains, evaluates, plots, and saves files.
- Inheritance creates coupled subclasses with no meaningful substitutability.

## Interview drill

Implement a class with `fit` and `predict`, add a fitted-state guard, test feature-count mismatch, and explain which fields are configuration versus learned state.

Executable reference: [OOP for ML](../../notebooks/01-foundations/03_python_oop_for_ml.ipynb). Reusable implementations: [`src/datacoding/algorithms`](../../src/datacoding/algorithms/).

## Connections

[[wiki/algorithms/from-scratch|From-scratch algorithms]] · [[wiki/workflows/sklearn-pipelines|sklearn estimator composition]] · [[wiki/workflows/pytorch-training-loop|`nn.Module` state]]
