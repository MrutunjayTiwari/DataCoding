---
type: map
status: active
tags: [libraries, map, python]
updated: 2026-08-30
---

# Library Map

Each library owns a different layer; confusion usually comes from crossing those boundaries implicitly.

| Library | Primary role | Core objects | Interview focus |
|---|---|---|---|
| NumPy | numerical representation and vectorized computation | `ndarray`, ufuncs, linear algebra | shape reasoning, broadcasting, indexing, distance math |
| pandas | labeled tabular manipulation | `Series`, `DataFrame`, groupby | joins, aggregation, reshaping, dates, leakage-safe features |
| scikit-learn | classical ML composition and evaluation | estimator, transformer, `Pipeline`, `ColumnTransformer` | preprocessing, cross-validation, metrics, search |
| PyTorch | differentiable models and training | `Tensor`, `Dataset`, `DataLoader`, `nn.Module` | autograd, model modes, loops, device and shape discipline |
| Seaborn | statistical visualization | axes-level and figure-level plotting functions | matching plot to question, labels, sampling, honest interpretation |
| AutoGluon | time-bounded automated model search | `TabularPredictor` | strong baseline, presets/time limit, held-out evaluation |
| PyCaret | concise experiment orchestration | `setup`, `compare_models`, `finalize_model`, `predict_model` | fast comparison, session state, careful holdout use |

## Data flow

`raw rows` → [[wiki/foundations/pandas|pandas]] → [[wiki/foundations/numpy|NumPy arrays]] → [[wiki/workflows/sklearn-pipelines|sklearn pipeline]] or [[wiki/workflows/pytorch-training-loop|PyTorch model]] → metrics/plots → optional [[wiki/workflows/automl|AutoML comparison]]

The boundary is not absolute: pandas is backed by arrays, sklearn accepts DataFrames, and PyTorch datasets may read DataFrames. The useful question is which layer owns the current transformation and whether that transformation is fitted only on training data.

## Connections

[[index|Home]] · [[maps/learning-path|Learning path]] · [[wiki/workflows/tabular-ml|Tabular ML]]
