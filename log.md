---
type: map
status: active
tags: [log, maintenance]
updated: 2026-09-06
---

# Activity Log

## [2026-09-06] refinement | pandas refresher rebuilt from reviewer notes

- Re-ingested the curated pandas recap against the 2026-09-05 review notes: added inspection, header/dtype repair, `as_index`, SQL-style ranks, lead/pct/expanding windows, time-based rolling with a validated key merge-back, calendar features, string operations, MultiIndex flattening, `concat` pitfalls, quick plots, a gotchas checklist, and drills; removed the inert `group_keys=False`, positional `.values` alignment, timezone and `Timestamp.now()` material, and the `floor("H")` alias that raises in pandas 3.
- Switched joins to the `pd.merge` function form and paired every window idiom with its index-aligned alternative; recorded each reviewer note and its decision in [[sources/notebook-fundamentals-audit|the notebook fundamentals audit]].
- Made dense preceding-line annotations the notebook standard in AGENTS.md (applied to the pandas refresher; NumPy and the remaining notebooks are a follow-up), documented the four notebook modes, the generator-first editing rule, and the `clear-output` tag; `run_notebooks.py` now clears tagged cells and merges stream chunks for stable diffs, and `check_vault.py` rejects embedded images.
- Verified execution under pandas 3.0.2, 2.2.3, and 2.1.4 with identical values; kept the `pandas>=2.1` floor because the PyCaret extra pins `pandas<2.2`.
- An independent adversarial review re-executed every cell, hand-checked the printed numbers, and corrected nine semantic statements (Copy-on-Write aliasing, chained-assignment warnings, offset-alias scope, `to_numeric` dtypes, `apply`/`transform` claims, MultiIndex assignment errors) before write-back; the NumPy refresher and the other notebooks still carry the earlier selective annotation style.

## [2026-09-05] refinement | Reader-driven notebook clarity

- Distilled reader feedback into durable rules for deterministic fixtures, local data flow, exact API semantics, single-purpose cells, and minimal presentation scaffolding.
- Reworked NumPy top-k to use a hand-checkable matrix and reorder the already-aligned candidate scores; split repeated-index accumulation, sliding windows, and NaN-aware reductions into separate demonstrations.
- Separated pandas missing-value repair from concatenation and PyTorch tensor operations from NumPy memory-sharing behavior.
- Removed custom display, seed, and project-root helpers plus presentation-only `to_string(...)` calls; used direct labeled output and moved a static AutoML checklist to Markdown.

## [2026-09-05] refinement | Annotation audit

- Re-audited all eleven notebooks against the intermediate-level annotation standard, including configuration and workflow boundaries missed by the first pass.
- Clarified display-only formatting, local RNG state, train-only statistics, stratification and scorer direction, gradient clearing and model modes, convolution shapes, figure lifecycle, and held-out AutoML reporting.
- Checked rendered comment placement after notebook formatting and retained conceptual explanations in Markdown when a line comment would become noisy.

## [2026-09-04] refinement | Revision annotations

- Reviewed all eleven notebooks for self-explanatory code at an intermediate interview-preparation level.
- Added selective comments for intent, shape/alignment changes, invariants, numerical stability, leakage boundaries, and subtle API behavior; kept obvious syntax uncommented.
- Reworked top-k annotations to distinguish unordered `argpartition` candidates, aligned gathers, candidate-only sorting, and mapping back to original positions.
- Made the annotation convention durable in the notebook generator and maintenance standard without adding concepts, outputs, or data.

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
