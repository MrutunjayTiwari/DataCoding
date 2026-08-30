---
type: workflow
status: active
tags: [sklearn, pipelines, preprocessing, cross-validation]
updated: 2026-08-30
---

# scikit-learn Pipelines

A pipeline makes the sequence of learned transformations part of the model and keeps cross-validation leakage-safe.

## Mental model

Every step before the final estimator implements `fit` plus `transform`; the final step implements `fit` plus `predict`. `ColumnTransformer` applies separate preprocessing to column groups and concatenates the results.

## Core workflow

1. Split rows before learning preprocessing statistics.
2. Identify numeric, nominal categorical, and genuinely ordinal columns.
3. Build small pipelines per column type.
4. Combine them with `ColumnTransformer`.
5. Add the estimator as the final `Pipeline` step.
6. Search nested parameters using `step__parameter` names.
7. Fit search on training data and evaluate once on untouched test data.

```python
pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", model),
    ]
)
```

## Encoder choices

- `OneHotEncoder(handle_unknown="ignore")` for unordered categories.
- `OrdinalEncoder(categories=[...], handle_unknown="use_encoded_value", unknown_value=-1)` only when order is meaningful.
- Tree models do not require scaling, but keeping preprocessing explicit makes model swaps safer.

## Failure modes

- Imputing or scaling the full dataset before cross-validation.
- Passing ordinal values through a nominal encoder, or inventing order for nominal values.
- Tuning on the held-out test set.
- Using accuracy by habit on imbalanced data.
- Unknown categories crash inference because encoder behavior was not specified.
- A tiny dataset uses more cross-validation folds than the minority class supports.

## Interview drill

Build a mixed-type classification and regression pipeline, inspect `get_params()` names, run a small randomized search, and explain exactly where leakage would occur if preprocessing sat outside the pipeline.

Executable reference: [Tabular pipelines](../../notebooks/03-sklearn/01_tabular_pipelines.ipynb).

## Connections

[[wiki/foundations/pandas|pandas]] · [[wiki/workflows/tabular-ml|Tabular ML]] · [[wiki/workflows/automl|AutoML]]

## Sources and provenance

Consolidates the curated sklearn quick-review notebook and the three legacy pipeline code printouts. The former binary-classification example evaluated on its training rows; the rebuilt notebook uses a held-out test set.
