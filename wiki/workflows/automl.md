---
type: workflow
status: active
tags: [automl, autogluon, pycaret]
updated: 2026-08-30
---

# AutoGluon and PyCaret

AutoML is a fast comparative baseline and experiment orchestrator, not a substitute for defining the target, split, metric, and leakage boundary.

## AutoGluon pattern

```python
predictor = TabularPredictor(label=target, eval_metric=metric, path=external_model_dir)
predictor.fit(train_data=train_df, time_limit=seconds, presets="medium_quality")
score = predictor.evaluate(test_df)
```

Keep predictor paths outside the vault. Pass a time limit in interview/practice environments and inspect the leaderboard rather than reporting only the winner.

Use the project's `autogluon` extra in its own environment under the external data root.

## PyCaret pattern

```python
experiment = setup(data=train_df, target=target, session_id=42, verbose=False)
best = compare_models()
final = finalize_model(best)
predictions = predict_model(final, data=test_df)
```

PyCaret uses experiment/session state, so avoid mixing classification and regression imports or datasets in the same uncontrolled namespace.

Use the project's `pycaret` extra in a separate environment. Do not combine it with the AutoGluon environment: their transitive NumPy, pandas, matplotlib, and model-library constraints can diverge.

## Fair comparison checklist

- Use the same train/test definition as the hand-built baseline.
- Choose the metric before seeing results.
- Keep the final test set outside model selection.
- Record time/compute budget and software versions.
- Inspect missing-value handling, label encoding, and failed models.
- Compare against a simple [[wiki/workflows/sklearn-pipelines|sklearn pipeline]].

## Failure modes

- Calling `finalize_model` before a true held-out evaluation and then reporting in-sample performance.
- AutoML output folders, logs, and model binaries enter the synced vault.
- Text/image tasks are forced through a tabular API without a clear representation.
- A failed notebook continues with stale variables from a previous experiment.
- API/version drift makes exploratory scratch cells unreliable.

## Interview drill

Explain how you would impose a five-minute budget, preserve a test set, choose a metric, compare the leaderboard winner with a simple baseline, and package inference.

Executable reference: [AutoML quickstart](../../notebooks/06-automl/01_automl_quickstart.ipynb). The notebook is guarded so the vault remains usable without either heavy optional stack and runs whichever stack is available in its active environment.

## Connections

[[wiki/workflows/tabular-ml|Tabular ML]] · [[wiki/workflows/sklearn-pipelines|sklearn pipelines]] · [[wiki/workflows/image-classification|Image classification]]

## Sources and provenance

Consolidates the useful portions of the legacy AutoGluon and PyCaret teaching notebooks. Error outputs, stale cross-experiment variables, portal smoke tests, and rough discovery cells were excluded.
