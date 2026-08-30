---
type: workflow
status: active
tags: [seaborn, visualization, eda]
updated: 2026-08-30
---

# Seaborn and Interview EDA

Choose a plot to answer a specific data question; do not generate a gallery and search for a story afterward.

## Question-to-plot map

| Question | Useful starting plot |
|---|---|
| What is one numeric distribution? | `histplot`, `ecdfplot`, `boxplot` |
| How does a numeric value vary by category? | `boxplot`/`violinplot` plus sampled points |
| How do two numeric variables relate? | `scatterplot`, optionally `regplot` |
| How do counts differ across categories? | `countplot` or an aggregated bar plot |
| How does a metric change over ordered time? | `lineplot` with explicit aggregation/error behavior |
| Where are correlations concentrated? | annotated `heatmap` on selected numeric columns |
| Do relationships vary by subgroup? | `FacetGrid`, `relplot`, or `catplot` |

## Core workflow

1. Confirm row meaning, dtype, missingness, and target distribution.
2. Compute a numerical summary before plotting.
3. Sample dense data deterministically for expensive plots.
4. Label title, axes, units, hue, and aggregation.
5. State what the plot cannot establish; association is not causation.

## Failure modes

- Pair plots become unreadable and expensive on many rows/features.
- A bar height silently represents a mean when the viewer expects a sum/count.
- Truncated axes exaggerate small differences.
- Overplotting hides density.
- Post-outcome variables are visualized as if they were valid predictors.
- Online sample datasets make a revision notebook fail offline.

## Interview drill

Given a mixed-type table, write one question per plot, produce a distribution, category comparison, relationship, and correlation view, then summarize each in one sentence.

Executable reference: [Seaborn interview EDA](../../notebooks/05-visualization/01_seaborn_interview_eda.ipynb), built on deterministic synthetic data with outputs cleared to keep the vault small.

## Connections

[[wiki/foundations/pandas|pandas]] · [[wiki/workflows/tabular-ml|Tabular ML]] · [[wiki/workflows/automl|AutoML]]
