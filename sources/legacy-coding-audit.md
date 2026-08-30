---
type: map
status: active
tags: [sources, migration, provenance]
updated: 2026-08-30
---

# Legacy Coding Source Audit

Source repository: `~/Desktop/geek/Coding` (preserved unchanged).

## Audit summary

- Approximately 108 MB across code, notebooks, PDFs, archives, logs, generated reports, and private/unrelated files.
- 121 notebook files, including checkpoints, exact duplicates, third-party tutorial collections, rough experiments, and curated teaching candidates.
- The new vault retains the underlying knowledge, not the source bloat.

## Retained through reconstruction

| Legacy material | New destination | Decision |
|---|---|---|
| NumPy refresher variants and advanced cells | [[wiki/foundations/numpy|NumPy]] + [notebook](../notebooks/01-foundations/01_numpy_interview_refresher.ipynb) | Consolidated the strongest version; labeled every retained output. |
| NumPy ML-from-scratch notebook | Algorithm pages, `src/datacoding/algorithms`, and algorithm notebooks | Removed rough cells, separated reusable classes, added KNN and tests. |
| Curated pandas recap | [[wiki/foundations/pandas|pandas]] + [notebook](../notebooks/01-foundations/02_pandas_interview_refresher.ipynb) | Reordered sections and made outputs traceable. |
| Curated sklearn pipeline notebook and pipeline code printouts | [[wiki/workflows/sklearn-pipelines|sklearn pipelines]] + [notebook](../notebooks/03-sklearn/01_tabular_pipelines.ipynb) | Unified regression/classification and corrected held-out evaluation. |
| PyTorch introduction, training drills, CIFAR case study, and portal dataset patterns | PyTorch/image workflow pages + two notebooks | Kept the reusable contracts; removed duplicate downloads and framework detours. |
| Curated Seaborn tutorial | [[wiki/workflows/visualization|Visualization]] + [notebook](../notebooks/05-visualization/01_seaborn_interview_eda.ipynb) | Rebuilt around interview questions and offline synthetic data; cleared plots. |
| AutoGluon/PyCaret teaching material | [[wiki/workflows/automl|AutoML]] + [notebook](../notebooks/06-automl/01_automl_quickstart.ipynb) | Kept stable workflows; excluded stale variables, errors, and discovery noise. |
| SQL PDF, Word refresher, SQL notebook, and strategy note | [[wiki/workflows/sql-interviews|SQL]] + [pattern library](../sql/interview_patterns.sql) | Converted to searchable Markdown/SQL; omitted bulky binary copies. |

## Excluded

- `.ipynb_checkpoints`, `.DS_Store`, logs, generated HTML reports, model weights, caches, and exact duplicates.
- Large third-party archive dumps and their extracted copies; these are better re-fetched from their original projects if needed.
- Rough notebooks with no teaching structure, stale API exploration, cross-cell state errors, or duplicate content already represented by a stronger source.
- General DSA/LeetCode material, by explicit scope decision.
- Private identity, administrative, and financial material. These remain only in the untouched legacy repository and must never be ingested into this vault.

## Data boundary

Datasets, downloaded weights, caches, and generated outputs live at `~/Desktop/geek/DataCoding-data` by default, resolved by `src/datacoding/config.py` and `DATACODING_DATA_DIR`.

## Connections

[[index|Home]] · [[maps/learning-path|Learning path]] · [[log|Activity log]]
