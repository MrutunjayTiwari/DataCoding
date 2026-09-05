---
type: map
status: active
tags: [audit, notebooks, fundamentals, curriculum]
updated: 2026-09-05
---

# Notebook Fundamentals Audit

This audit checks whether each notebook teaches the small set of operations a learner should be able to reconstruct under interview pressure. A topic qualifies as a must-have when it is frequent, failure-prone, transferable across tasks, and compact enough for retrieval practice. The goal is coverage, not an API catalog.

## Findings and actions

| Notebook | Must-have bar | Audit finding | Action |
|---|---|---|---|
| NumPy refresher | creation/dtypes, shape operations, combine/split, indexing, broadcasting, reductions, ranking, vectorization, linear algebra, numerical stability | Combining arrays was missing; singleton-axis and `where(condition)` examples had been added without durable generator coverage or labeled output. | Added `concatenate` versus `stack`, `vstack`/`hstack`, `split`/`array_split`, metadata/casting, and clean singleton-axis/`where` demonstrations. |
| pandas refresher | inspect, filter/assign, missingness/dtypes, combine, group, time operations, reshape, join | Missing-data repair and `pd.concat` were absent; compound masks were only implicit elsewhere. | Added a compact dirty-column repair, missingness audit, row concatenation, compound mask, and assertions. |
| Python OOP for ML | constructor configuration, learned state, methods, composition, fitted guards, boundary errors | Meets the ML-focused OOP bar. General inheritance hierarchies would dilute the estimator lesson. | No expansion. |
| Linear models | train-only scaling, held-out evaluation, linear/logistic/perceptron behavior, optimization check | Gradient descent was covered, but linear regression lacked a direct least-squares reference. | Added `np.linalg.lstsq` on the same training split and a held-out comparison. |
| KNN and K-means | pairwise distances, voting, scale sensitivity, assignment/update, convergence state, edge checks | Core mechanics are explicit and tested. Distance weighting and restart selection are useful extensions, not minimum recall. | No expansion. |
| sklearn pipelines | split, trivial baseline, mixed-type preprocessing, leakage-safe CV/search, held-out metrics, feature inspection | Pipelines/search were strong, but no executable dummy baseline established whether modeling added value. | Added `DummyClassifier`/`DummyRegressor` comparisons and clarified when random splitting is appropriate. |
| PyTorch fundamentals | tensor construction/interchange, shape operations, `cat`/`stack`, autograd, loader, module, train/eval, device, checkpoint | Training was covered well; basic tensor reshaping, combining, and the NumPy boundary were too thin. | Added `unsqueeze`, `permute`, `cat`, `stack`, `from_numpy` sharing, copy semantics, and `detach().cpu().numpy()`. |
| Image classification | dataset/label contract, NCHW, logits/loss, split discipline, train/eval, class-level evaluation, ID-keyed inference | End-to-end flow was sound, but aggregate accuracy alone did not enact the documented class-imbalance warning. | Added confusion matrix and per-class recall without adding files or plots. |
| Seaborn EDA | distribution, category comparison, relationship/faceting, selected correlations, numerical interpretation | The four high-value question families are covered. More plot types would turn it into a gallery. | No expansion. |
| AutoML quickstart | held-out set, metric/budget, external artifacts, leaderboard/evaluation, environment guard | Appropriate for an optional heavy workflow. Actual training remains opt-in and isolated by environment. | No expansion. |
| Atlassian revision | explicit contract, robust implementation, edge tests, complexity, scale redesign, interview communication | Correctly specialized around the recruiting guidance and already reuses core NumPy/pandas patterns. | No duplicated foundation material. |

## Cross-cutting coverage decisions

- PCA remains a wiki page plus a reusable tested estimator and SVD practice in NumPy. A separate notebook would duplicate mechanics without adding a new interview workflow.
- SQL remains a searchable `.sql` pattern library. Wrapping the same queries in a notebook would add execution infrastructure rather than knowledge.
- Data-file I/O is documented on [[wiki/foundations/pandas|pandas]], but the quick notebook stays in-memory so it is deterministic and cloud-safe. Real projects should parameterize paths under the external data root.
- Grouped and chronological splitting, threshold selection, probability calibration, real image decoding/augmentation, transfer learning, mixed precision, and full AutoML runs are context-dependent workflow extensions. Their governing principles are documented, but they are not forced into every quick notebook.
- Additional from-scratch families such as decision trees and Naive Bayes are curriculum expansions, not missing primitives inside the notebooks audited here. Add them when a target role or interview makes them valuable rather than crowding the current revision path.
- Exhaustive constructor lists, pandas styling, row-wise `apply` tours, deep inheritance examples, model zoos, and plot galleries are intentionally excluded. They are lookup material or encourage weak habits rather than retrieval-worthy fundamentals.

## Compactness result

The audit adds no new teaching notebook. It strengthens six existing notebooks in place, preserving the layered design: concepts in the wiki, reusable algorithms in `src/`, and short executable drills in notebooks.

## Revision annotation standard

All eleven notebooks use selective code comments as retrieval cues. A comment earns space when it explains intent, an important shape or alignment change, an invariant, numerical stability, leakage control, or behavior that is easy to misremember. Display/runtime configuration is annotated when it could be mistaken for a data transformation—for example, `np.set_printoptions` explicitly says that array values are unchanged. Comments use parameterized language such as “keep `k` candidates” rather than example-specific wording such as “keep two.” Obvious imports, assignments, and syntax remain uncommented; longer conceptual explanations stay in the preceding Markdown cell, and runtime evidence stays in labeled output.

## Connections

[[index|Home]] · [[maps/learning-path|Learning path]] · [[maps/interview-revision|Interview revision]] · [[sources/legacy-coding-audit|Legacy source audit]]
