---
type: map
status: active
tags: [log, maintenance]
updated: 2026-08-30
---

# Activity Log

## [2026-09-04] audit | Notebook fundamentals

- Audited all eleven notebooks against compact, interview-oriented must-have checklists and recorded decisions in [[sources/notebook-fundamentals-audit|Notebook fundamentals audit]].
- Added array combining/splitting and cleaned the pulled singleton-axis/`where` additions in the NumPy refresher while preserving their teaching intent.
- Added pandas missing-data/dtype repair and row concatenation, a linear least-squares reference, sklearn dummy baselines, PyTorch tensor-shape/interchange patterns, and image-classification confusion/per-class metrics.
- Deliberately avoided new notebooks, API catalogs, model/plot galleries, duplicated PCA/SQL material, and context-dependent advanced workflows.

## [2026-08-30] ingest | Atlassian ML coding guidance

- Reviewed the diarized recruiter conversation as a single immutable source and separated confirmed round guidance from candidate assumptions.
- Added a quick, self-contained [Atlassian revision notebook](notebooks/07-interview-specific/01_atlassian_notebook.ipynb) focused on weighted sampling, coupon recommendation, tests, scale, and interview communication.
- Kept the core to standard Python and NumPy; included pandas only as an explicitly optional refresher because library expectations were not confirmed.
- Recorded the source decision in [[sources/atlassian-ml-coding-guidance|Atlassian ML coding guidance]] and linked the notebook from the home and interview revision maps.

## [2026-08-30] ingest | Legacy Coding folder

- Inventoried the legacy repository and reviewed candidate notebooks, Markdown, SQL references, PDFs, and the Word refresher.
- Established a new ML/data-focused vault; general DSA remains out of scope.
- Preserved the legacy repository unchanged and excluded private/unrelated documents, transaction data, archives, logs, checkpoints, generated reports, and model files.
- Rebuilt the high-signal material as linked wiki pages, annotated notebooks, reusable implementations, and tests.
- Routed data, model weights, caches, and generated outputs to `$HOME/Desktop/geek/DataCoding-data`.
- Executed all six quick notebooks and all four optional/heavy notebooks; retained only compact quick-study outputs.
- Verified the reusable algorithms with ten tests and checked links, provenance, notebook contracts, output size, formatting, and vault boundaries.
- Split AutoGluon and PyCaret into separately resolvable external environments so their dependency constraints cannot destabilize the core environment.
