---
type: map
status: active
tags: [interviews, revision, drills]
updated: 2026-09-04
---

# Interview Revision Map

Use retrieval practice: attempt first, inspect the reference second, then rewrite the weakest part from memory.

## Thirty-minute emergency pass

1. Predict `concatenate`, `stack`, broadcasting, and distance shapes in the [NumPy refresher](../notebooks/01-foundations/01_numpy_interview_refresher.ipynb).
2. Rehearse missing-value repair, `concat`, `groupby().agg`, `transform`, `merge`, and time-safe `shift` in the [pandas refresher](../notebooks/01-foundations/02_pandas_interview_refresher.ipynb).
3. Recite the [[wiki/workflows/sklearn-pipelines|pipeline leakage rules]].
4. Write the canonical [[wiki/workflows/pytorch-training-loop|PyTorch train/eval loop]] on paper.
5. Review `ROW_NUMBER`, `LAG`, and explicit window frames in [[wiki/workflows/sql-interviews|SQL interview patterns]].

## Atlassian ML coding pass

Use the [Atlassian notebook](../notebooks/07-interview-specific/01_atlassian_notebook.ipynb) to rehearse the two examples named in the recruiting guidance: weighted sampling and coupon recommendation. The short pass emphasizes explicit contracts, NumPy implementation patterns, edge-case tests, complexity, and redesign at larger scale. Treat pandas as optional insurance because the expected library set was not confirmed.

Source decision: [[sources/atlassian-ml-coding-guidance|Atlassian ML coding guidance]].

## Two-hour pass

- Implement one linear model and one distance-based algorithm from scratch.
- Build a mixed numeric/categorical sklearn pipeline.
- Explain metric choice for one regression and one imbalanced classification task.
- Sketch an image-classification data flow from file paths to predictions.

## Seven-session rotation

| Session | Retrieval target | Executable reference |
|---|---|---|
| 1 | shapes, combining, axes, broadcasting | [NumPy](../notebooks/01-foundations/01_numpy_interview_refresher.ipynb) |
| 2 | missingness, concat, groupby, joins, dates | [pandas](../notebooks/01-foundations/02_pandas_interview_refresher.ipynb) |
| 3 | linear/logistic/perceptron | [Linear models](../notebooks/02-algorithms-from-scratch/01_linear_models.ipynb) |
| 4 | KNN/K-means | [Neighbors/clustering](../notebooks/02-algorithms-from-scratch/02_knn_kmeans.ipynb) |
| 5 | baselines, preprocessing, and model search | [sklearn](../notebooks/03-sklearn/01_tabular_pipelines.ipynb) |
| 6 | autograd and loops | [PyTorch](../notebooks/04-pytorch/01_pytorch_fundamentals.ipynb) |
| 7 | EDA, SQL, and an end-to-end verbal walkthrough | [Seaborn](../notebooks/05-visualization/01_seaborn_interview_eda.ipynb) |

## Verbal walkthrough checklist

1. Restate target, unit of observation, and constraints.
2. Name input/output shapes and split strategy.
3. Establish a simple baseline and metric.
4. Identify leakage and edge cases before optimization.
5. Implement in small testable pieces.
6. Inspect errors, not only the aggregate score.
7. State complexity and the next improvement.

## Connections

[[index|Home]] · [[maps/learning-path|Learning path]] · [[wiki/algorithms/from-scratch|From-scratch map]]
