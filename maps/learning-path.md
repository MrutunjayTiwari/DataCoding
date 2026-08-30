---
type: map
status: active
tags: [learning-path, curriculum, ml]
updated: 2026-08-30
---

# Learning Path

Move from array mechanics to complete ML systems without skipping the contracts between layers.

## Stage 1 - Numerical Python

1. [[wiki/foundations/numpy|NumPy]]: predict shapes before running code; practice broadcasting, boolean masks, reductions, and vectorized distance calculations.
2. [[wiki/foundations/pandas|pandas]]: select, group, reshape, join, and build time-aware features without leakage.
3. [[wiki/foundations/python-oop|Python OOP for ML]]: understand constructor parameters, learned attributes, `fit`, `predict`, composition, and tests.

**Exit check:** implement standardization, pairwise squared distances, and a groupwise top-k query without looking up syntax.

## Stage 2 - Algorithms from scratch

Follow [[wiki/algorithms/from-scratch|the from-scratch map]] in this order:

1. [[wiki/algorithms/linear-regression|Linear regression]]
2. [[wiki/algorithms/logistic-regression|Logistic regression]]
3. [[wiki/algorithms/perceptron|Perceptron]]
4. [[wiki/algorithms/knn|K-nearest neighbors]]
5. [[wiki/algorithms/kmeans|K-means]]
6. [[wiki/algorithms/pca|PCA via SVD]]

**Exit check:** state shapes and objective, implement `fit`/`predict`, test an edge case, and compare behavior with a trusted implementation.

## Stage 3 - Library workflows

- [[wiki/workflows/sklearn-pipelines|scikit-learn pipelines]] for leakage-safe preprocessing and search.
- [[wiki/workflows/tabular-ml|Tabular ML]] for regression/classification problem framing and evaluation.
- [[wiki/workflows/visualization|Seaborn and interview EDA]] for question-driven plots.
- [[wiki/workflows/sql-interviews|SQL patterns]] for analytical data extraction.

**Exit check:** build one `ColumnTransformer` pipeline and explain why all learned preprocessing must be inside cross-validation.

## Stage 4 - PyTorch

- [[wiki/workflows/pytorch-training-loop|Training loop]]: tensor shapes, autograd, model modes, loss aggregation, and evaluation.
- [[wiki/workflows/image-classification|Image classification]]: dataset contracts, transforms, CNNs, transfer learning, and inference.

**Exit check:** write a full train/eval loop from memory and debug a deliberate shape mismatch.

## Stage 5 - End-to-end speed

- Use [[wiki/workflows/automl|AutoGluon or PyCaret]] only after establishing a hand-built baseline.
- Rehearse complete tabular and image tasks under time limits.
- Use [[maps/interview-revision|Interview revision map]] for spaced repetition.

## Connections

[[index|Home]] · [[maps/library-map|Library map]] · [[maps/interview-revision|Interview revision]]
