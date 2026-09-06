---
type: map
status: active
tags: [audit, notebooks, fundamentals, curriculum]
updated: 2026-09-06
---

# Notebook Fundamentals Audit

This audit checks whether each notebook teaches the small set of operations a learner should be able to reconstruct under interview pressure. A topic qualifies as a must-have when it is frequent, failure-prone, transferable across tasks, and compact enough for retrieval practice. The goal is coverage, not an API catalog.

## Findings and actions

| Notebook | Must-have bar | Audit finding | Action |
|---|---|---|---|
| NumPy refresher | creation/dtypes, shape operations, combine/split, indexing, broadcasting, reductions, ranking, vectorization, linear algebra, numerical stability | Combining arrays was missing; singleton-axis and `where(condition)` examples had been added without durable generator coverage or labeled output. | Added `concatenate` versus `stack`, `vstack`/`hstack`, `split`/`array_split`, metadata/casting, and clean singleton-axis/`where` demonstrations. |
| pandas refresher | inspect, select/assign, header and dtype repair, missingness, dedup/top-k/ranks, `agg` variants and `transform`, row/time/cumulative windows, dates, strings, reshape with MultiIndex flattening, `pd.merge`/`concat` | 2026-09-05 review: the compact rebuild had dropped curated-copy material (inspection, header/dtype repair, `contains`/`split`/`replace`, `as_index`, ranks, lead/pct/expanding windows, time-based rolling, MultiIndex flattening, `concat` pitfalls, quick plots, gotchas) and carried an inert `group_keys=False` on a `head()` chain. | Rebuilt as a 23-code-cell refresher on one shared fixture with dense annotations; every reviewer note resolved in the table below. |
| Python OOP for ML | constructor configuration, learned state, methods, composition, fitted guards, boundary errors | Meets the ML-focused OOP bar. General inheritance hierarchies would dilute the estimator lesson. | No expansion. |
| Linear models | visible NumPy implementations, train-only scaling, held-out evaluation, linear/logistic/perceptron behavior, optimization check | The notebook imported all three algorithms from the local package, so it was neither standalone nor useful for revising the implementations; linear regression also lacked a direct least-squares reference. | Defined concise linear-regression, logistic-regression, and perceptron classes in the notebook, removed project imports and setup helpers, and retained `np.linalg.lstsq` as a held-out reference. The hardened `src/` implementations remain the tested canonical versions. |
| KNN and K-means | pairwise distances, voting, scale sensitivity, assignment/update, convergence state, edge checks | Core mechanics are explicit and tested. Distance weighting and restart selection are useful extensions, not minimum recall. | No expansion. |
| sklearn pipelines | split, trivial baseline, mixed-type preprocessing, leakage-safe CV/search, held-out metrics, feature inspection | Pipelines/search were strong, but no executable dummy baseline established whether modeling added value. | Added `DummyClassifier`/`DummyRegressor` comparisons and clarified when random splitting is appropriate. |
| PyTorch fundamentals | tensor construction/interchange, shape operations, `cat`/`stack`, autograd, loader, module, train/eval, device, checkpoint | Training was covered well; basic tensor reshaping, combining, and the NumPy boundary were too thin. | Added `unsqueeze`, `permute`, `cat`, `stack`, `from_numpy` sharing, copy semantics, and `detach().cpu().numpy()`. |
| Image classification | dataset/label contract, NCHW, logits/loss, split discipline, train/eval, class-level evaluation, ID-keyed inference | End-to-end flow was sound, but aggregate accuracy alone did not enact the documented class-imbalance warning. | Added confusion matrix and per-class recall without adding files or plots. |
| Seaborn EDA | distribution, category comparison, relationship/faceting, selected correlations, numerical interpretation | The four high-value question families are covered. More plot types would turn it into a gallery. | No expansion. |
| AutoML quickstart | held-out set, metric/budget, external artifacts, leaderboard/evaluation, environment guard | Appropriate for an optional heavy workflow. Actual training remains opt-in and isolated by environment. | No expansion. |
| Atlassian revision | explicit contract, robust implementation, edge tests, complexity, scale redesign, interview communication | Correctly specialized around the recruiting guidance and already reuses core NumPy/pandas patterns. | No duplicated foundation material. |

## 2026-09-06 pandas refresher re-ingest

Source: the curated recap `~/Desktop/geek/Coding/old_data_coding/curated_copy/Pandas_Interview_Recap_MiniNotebook.ipynb` (immutable) re-read against the reviewer's 2026-09-05 notes on the compact rebuild. The reviewer judged the curated copy easier to revise and more comprehensive, although dirty; the rebuild keeps its coverage, removes the hairy bits, and adds the annotation density the reviewer asked for.

| Reviewer note | Decision |
|---|---|
| `group_keys=False` unannotated on the top-2 chain | Removed from the `head()` chain, where it has no effect (filters keep the original index). Demonstrated where it matters: the index of `apply` output, with `head()` shown unchanged alongside. |
| Can the rolling mean avoid `reset_index`? | Yes: `transform(lambda s: s.rolling(k, min_periods=1).mean())` keeps the original index. Shown next to `groupby().rolling()` + `reset_index(level=0, drop=True)` with an assertion that both agree and a print of the raw `(user_id, index)` MultiIndex. |
| `contains` and other string operations missing | Strings section: `contains(case=False, na=False)`, `extract(expand=False)`, `split(n=1, expand=True)`, `replace(regex=True)`, `len`, chained cleanup, nullable `Int64` for extracted digits. |
| Pivot example oversimplified; MultiIndex handling missing | `pivot` failure on duplicate pairs, single-level `pivot_table`, multi-level `pivot_table` flattened with `"_".join(map(str, col))`; dict-of-lists `agg` flattening in the groupby section; `melt` round trip. |
| `as_index` not taught | Named aggregation with `as_index=False` beside the default indexed form, plus `dropna=False` for missing keys. |
| Prefer `pd.merge` | Function form used throughout and recorded as a standard in AGENTS.md. |
| Easier to revise, more comprehensive | One shared fixture with a deliberate revenue tie and a missing note; one job per cell; the Markdown above each cell states the question, the failure mode, and the API semantics at the point of risk; dense preceding-line comments; gotchas checklist; drills; 20-minute pass. |
| Remove the truly hairy bits | Dropped positional `.values` alignment (replaced by a validated key merge), timezone conversion, `Timestamp.now()` (fixed cutoff), `floor("H")` (raises in pandas 3; now `"h"`), and the unlabeled `df.info()` block. Quick `df.plot` kept in a `clear-output` cell. |

Verification: executed under pandas 3.0.2, 2.2.3, and 2.1.4 with identical values (only `None`/`NaN`, `object`/`str`, and `[ns]`/`[us]` display differences), so the `pandas>=2.1` floor stands and the PyCaret environment's `pandas<2.2` pin remains installable. An independent adversarial review then re-executed every cell, hand-checked the printed numbers against the fixture, and corrected nine semantic statements (Copy-on-Write aliasing, chained-assignment warnings, the scope of lowercase offset aliases, `to_numeric` output dtypes, `apply` versus `transform` claims, and the `TypeError` raised when a `(key, index)`-indexed result is assigned to a flat frame) before write-back.

## Cross-cutting coverage decisions

- PCA remains a wiki page plus a reusable tested estimator and SVD practice in NumPy. A separate notebook would duplicate mechanics without adding a new interview workflow.
- SQL remains a searchable `.sql` pattern library. Wrapping the same queries in a notebook would add execution infrastructure rather than knowledge.
- Data-file I/O is documented on [[wiki/foundations/pandas|pandas]], but the quick notebook stays in-memory so it is deterministic and cloud-safe. Real projects should parameterize paths under the external data root.
- Grouped and chronological splitting, threshold selection, probability calibration, real image decoding/augmentation, transfer learning, mixed precision, and full AutoML runs are context-dependent workflow extensions. Their governing principles are documented, but they are not forced into every quick notebook.
- Additional from-scratch families such as decision trees and Naive Bayes are curriculum expansions, not missing primitives inside the notebooks audited here. Add them when a target role or interview makes them valuable rather than crowding the current revision path.
- Exhaustive constructor lists, pandas styling, row-wise `apply` tours, deep inheritance examples, model zoos, and plot galleries are intentionally excluded. They are lookup material or encourage weak habits rather than retrieval-worthy fundamentals.
- Timezone conversion and third-party pandas tutorials in the legacy archive remain excluded; the refresher covers naive timestamps only.

## Compactness result

The 2026-09-04 audit added no new teaching notebook and strengthened six existing notebooks in place. The 2026-09-06 rebuild is the deliberate exception: the pandas refresher grew from 11 to 23 code cells because the reviewer judged the compact version under-comprehensive for revision. The layered design is unchanged: concepts in the wiki, reusable algorithms in `src/`, and executable drills in notebooks.

## Revision annotation standard

Every consequential statement is preceded by a short comment stating its intent, the shape or alignment it changes, the invariant it relies on, or the API behavior that is easy to misremember. Comments use parameterized language such as "keep `k` candidates" rather than example-specific wording such as "keep two". The comment sits on the line above the statement so the formatter never wraps code around a long trailing comment; a trailing comment is acceptable only when code plus comment fit the line length. Display/runtime configuration is annotated when it could be mistaken for a data transformation—for example, `pd.set_option("display.precision", 3)` explicitly says that values are unchanged. Obvious imports, assignments, and prints remain uncommented; the Markdown above each cell names the question the cell answers and the failure mode it prevents, and runtime evidence stays in labeled output.

The pandas refresher applies this standard in full. The other ten notebooks still carry the earlier selective style (intent, shape, invariant, and subtle-API comments only) and are scheduled to be re-annotated notebook by notebook, NumPy first.

## Revision clarity standard

Reader feedback exposed six ways a technically correct notebook can still be awkward to revise:

- Use a small deterministic fixture when the lesson is the operation rather than randomness. A monotonic score matrix makes top-k alignment verifiable by eye; one shared event table with a planted tie and a planted missing value makes ranking and text handling verifiable the same way.
- Preserve local data flow. Once candidate scores have been gathered, reorder those scores with `candidate_order` rather than gathering again from the full source matrix.
- State exact API semantics at the point of risk. `kth` is a zero-based position, negating scores reverses which original values occupy the smallest partition positions, `argpartition` does not guarantee order within the selected candidates, and `group_keys` changes only the index of `apply` output.
- Give each cell one coherent learning job. Repeated-index accumulation, sliding windows, and NaN-aware reduction deserve separate cells because they answer unrelated questions; so do `agg` variants, `transform`, and `group_keys`.
- Remove presentation scaffolding from study code. Direct labeled `print(...)` calls are closer to interview code than custom display helpers or repeated `to_string(...)` conversions.
- Prefer direct assignment when a method chain or lambda adds syntax but does not teach a distinct operation; retain idioms such as `groupby`, `pd.merge`, masks, and pipelines when those are the material being revised.
- Make the implementation visible when the notebook promises "from scratch." It may use NumPy, but it must not require an installed copy of this project or hide the algorithm behind a package import; concise class methods are lesson content, whereas setup/display helpers are boilerplate.

These are editing rules, not reasons to add more material. An extra intermediate, comparison, or Markdown sentence should earn its place by making a failure-prone idea reconstructable under interview pressure.

## Connections

[[index|Home]] · [[maps/learning-path|Learning path]] · [[maps/interview-revision|Interview revision]] · [[sources/legacy-coding-audit|Legacy source audit]]
