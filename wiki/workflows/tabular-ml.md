---
type: workflow
status: active
tags: [tabular, classification, regression, evaluation]
updated: 2026-08-30
---

# Tabular Regression and Classification

The hard part of tabular ML is defining the row, target, split, and leakage boundary before choosing a model.

## Problem contract

Write down:

- unit of observation;
- target and prediction horizon;
- information available at prediction time;
- group/time dependencies between rows;
- operational cost of each error type;
- final evaluation metric.

## Baseline workflow

1. Audit target distribution, duplicates, missingness, cardinality, and dates.
2. Choose random, stratified, grouped, or chronological splitting based on deployment.
3. Establish a dummy/simple linear or tree baseline.
4. Put learned preprocessing into [[wiki/workflows/sklearn-pipelines|a pipeline]].
5. Tune on training folds only.
6. Evaluate once on test data and inspect segment-level errors.
7. Save generated models/reports under the external data root, never in the vault.

## Classification metrics

- Accuracy: useful only when classes and error costs are reasonably balanced.
- Precision/recall/F1: thresholded behavior for the positive class.
- ROC AUC: ranking across thresholds; can look optimistic with severe imbalance.
- Average precision/PR AUC: emphasizes positive-class retrieval.
- Log loss: probability quality and confidence.

## Regression metrics

- MAE: typical absolute error, robust relative to squared loss.
- RMSE: penalizes large errors more strongly.
- R²: improvement relative to predicting the mean; not an error unit.

## Failure modes

- Entity leakage across train/test rows.
- Future information or target-derived fields in predictors.
- Random split for temporal deployment.
- High-cardinality IDs memorized as features.
- Aggregate score hides poor performance for a critical subgroup.
- Preprocessing/model search silently sees test data.

## Interview drill

Given a table and business prompt, explain the split and metric before writing model code. Then build both a classification and regression baseline from the same mixed-type feature schema.

Executable reference: [Tabular pipelines](../../notebooks/03-sklearn/01_tabular_pipelines.ipynb).

## Connections

[[wiki/foundations/pandas|pandas]] · [[wiki/workflows/sklearn-pipelines|sklearn pipelines]] · [[wiki/workflows/automl|AutoML]] · [[wiki/workflows/visualization|EDA]]
