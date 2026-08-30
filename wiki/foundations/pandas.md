---
type: concept
status: active
tags: [pandas, tabular, data-wrangling]
updated: 2026-08-30
---

# pandas

pandas is the labeled-data layer between raw records and model-ready arrays.

## Mental model

A `DataFrame` has two axes and an index. Most interview operations fall into five verbs: select, assign, aggregate, reshape, and combine. State the unit represented by one row before doing any of them.

## Core patterns

### Selection and safe assignment

- `loc` selects by labels or boolean conditions.
- `iloc` selects by integer position.
- Assign through `df.loc[mask, "column"] = value`; avoid modifying an ambiguous slice.

### Aggregation versus transform

`groupby().agg(...)` reduces groups to fewer rows. `groupby().transform(...)` returns one value per original row, making it suitable for group-normalized features and comparisons.

### Window features

Sort explicitly before `shift`, `rolling`, or cumulative operations. When predicting the current row, lag a feature before rolling if the current value would leak target-time information.

### Reshaping

- `pivot` requires unique index/column pairs.
- `pivot_table` aggregates duplicates.
- `melt` turns repeated wide columns into tidy key/value rows.
- Flatten MultiIndex columns immediately after multi-aggregation when downstream code expects strings.

### Joins

Before `merge`, state expected row cardinality and validate it with `validate="many_to_one"`, `"one_to_one"`, etc. Check unmatched keys with `indicator=True` when correctness matters.

## Failure modes

- Duplicate keys silently multiply rows during a join.
- A global preprocessing step learns from the test period.
- Dates remain strings, producing lexical rather than chronological order.
- `groupby` drops missing keys unless `dropna=False` is chosen deliberately.
- `inplace=True` obscures data flow and does not guarantee lower memory use.
- A printed table has no label identifying the transformation that produced it.

## Interview drill

Given an event table, compute per-user top-2 revenue events, prior-event time gaps, a lagged rolling mean, a wide summary, and a validated customer join.

Executable reference: [pandas interview refresher](../../notebooks/01-foundations/02_pandas_interview_refresher.ipynb).

## Connections

[[wiki/foundations/numpy|NumPy]] · [[wiki/workflows/sklearn-pipelines|sklearn pipelines]] · [[wiki/workflows/tabular-ml|Tabular ML]] · [[wiki/workflows/sql-interviews|SQL]]

## Sources and provenance

Compiled from the curated pandas recap notebook, with section order repaired and output labels added.
