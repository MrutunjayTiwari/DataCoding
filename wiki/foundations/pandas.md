---
type: concept
status: active
tags: [pandas, tabular, data-wrangling]
updated: 2026-09-06
---

# pandas

pandas is the labeled-data layer between raw records and model-ready arrays.

## Mental model

A `DataFrame` has two axes and an index. Most interview operations fall into six verbs: inspect, select/assign, aggregate, window, reshape, and combine. State the unit represented by one row before doing any of them, and remember that almost every operation aligns on the index, not on position.

## Core patterns

### Inspect first

`shape`, `dtypes`, `isna().sum()`, `nunique()`, `head()`/`sample(random_state=...)`, and `describe()` establish the grain, the gaps, and the cardinality before any transformation. A candidate key has `nunique() == len(df)`.

### Selection and assignment

- `loc` selects by labels or boolean masks; label slices are inclusive. `iloc` selects by integer position; the end is exclusive.
- Parenthesize each condition around `&`/`|`; `isin` and `between` produce masks.
- pandas 3 enables Copy-on-Write: a derived frame (`df[:]`, `df[cols]`, `df[mask]`) never writes back to its parent, and chained assignment `df[mask]["col"] = v` never reaches the parent (pandas 3 warns with `ChainedAssignmentError`; 2.x warned with `SettingWithCopyWarning`, and the reversed form `df["col"][mask] = v` could write through there). A plain alias (`other = df`) is the same object and does write through. Either way, assign through `df.loc[mask, "column"] = value` on the frame you intend to change.

### Missing data and dtype repair

Normalize headers once (`df.columns.str.strip().str.lower().str.replace(r"[^0-9a-z]+", "_", regex=True)`), then coerce: `pd.to_numeric(..., errors="coerce")` and `pd.to_datetime(..., errors="coerce")` turn unparseable values into `NaN`/`NaT` so the damage is countable (a coerced NaN makes the result `float64`; all-parseable text gives `int64`; nullable `string` input yields `Int64`/`<NA>`). `astype("category")` stores repeated labels as codes; a categorical accepts only existing levels in `fillna`, so add the level first. Learn fill values on training rows only, drop rows whose key fields cannot be repaired, and do not invent labels by imputing a missing supervised target. `count()` and `value_counts()` skip missing values; audits use `size` and `dropna=False`.

### Sorting, deduplication, ranks

Sort with an explicit tie-breaker column so results are deterministic. `drop_duplicates(keep="last")` after sorting gives the latest row per key. `groupby().head(k)` keeps k rows per group on the original index; dedupe values first when k must count distinct values. `cumcount()` is `ROW_NUMBER`, `rank(method="dense")` is `DENSE_RANK`, and `rank(method="min")` is `RANK` with gaps.

### Aggregation versus transform

`groupby().agg(name=(column, function))` reduces groups to fewer rows with flat headers; `as_index=False` keeps the key as a column. The dict-of-lists form creates `(column, function)` MultiIndex headers: flatten them with `"_".join(col)`. `groupby().transform(...)` returns one value per original row, making it suitable for group-normalized features and row-aligned comparisons. `group_keys` affects only the index of `apply` output; prefer `agg`/`transform` with built-in function names, which run in compiled code and have a fixed output shape. Missing keys are dropped unless `dropna=False`.

### Window features

Sort explicitly before `shift`, `rolling`, `expanding`, or cumulative operations, and compute inside `groupby(key)` so windows never cross entities. `shift(1)` is `LAG`, `shift(-1)` is `LEAD`. Row windows (`rolling(3)`) count rows; time windows (`rolling("3D")`) count days on a `DatetimeIndex` or `on=` column and come back keyed by `(group, time)`, so join them back on those keys rather than aligning positionally. `transform(lambda s: s.rolling(k).mean())` keeps the original index; `groupby().rolling()` is faster but returns a `(key, index)` MultiIndex to drop first. Lag before rolling when the feature must exclude the current row.

### Dates

`.dt.normalize()` keeps a midnight `Timestamp`; `.dt.date` gives Python dates; `floor("h")` truncates; `to_period("W")`/`("M")` are calendar buckets, not rolling windows; `dayofweek` is Monday=0. Offset aliases: sub-daily ones are lowercase (`h`, `min`, `s`), and `floor`/`round`/`ceil` accept only those fixed frequencies plus `D`; month/quarter/year-end offsets are `ME`/`QE`/`YE` for `resample`, `date_range`, and `Grouper` (`H` and `M` raise in pandas 3), while `to_period` keeps `M`/`Q` and `D`/`W` are unchanged. Aggregating by `groupby(ts.dt.normalize())` drops days with no rows; `resample("D")` on a `DatetimeIndex` or `on=` column, or `pd.Grouper(key=..., freq="D")`, keeps them as rows (0 for `sum`/`count`/`nunique`, NaN for `mean`). Comparisons with `NaT` are False, so date filters drop unparseable rows automatically.

### Strings

`str.contains(pattern, case=False, na=False)`, `str.extract(r"(...)", expand=False)`, `str.split(sep, n=1, expand=True)`, `str.replace(r"\D+", "", regex=True)`, and `str.len()` cover most text questions. Cast extracted digits to nullable `Int64` so a missing value stays `<NA>` instead of forcing floats.

### Reshaping

- `pivot` requires unique index/column pairs and raises otherwise.
- `pivot_table` aggregates duplicates; lists in `values` or `aggfunc` create multi-level headers to flatten with `"_".join(map(str, col))`.
- `melt` turns repeated wide columns into tidy key/value rows; `pivot` inverts it when the pairs are unique.

### Joins and concatenation

Use `pd.merge(left, right, on=..., how=..., validate="many_to_one", indicator=True, suffixes=(...))`: `validate` raises when a lookup key would multiply rows, `indicator` records match provenance, `suffixes` resolves overlapping non-key columns, and `left_on`/`right_on` handle differing key names. An outer join with the indicator audits unmatched keys. `pd.concat(axis=0, ignore_index=True)` appends row batches; `axis=1` aligns on index labels, so reset both indexes for positional side-by-side placement.

### I/O boundary

For real data, make paths parameters under the external data root. With `read_csv`, specify important dtypes/date parsing when known and inspect shape, dtypes, missingness, duplicates, and a small sample immediately. The quick notebook stays in-memory so it never depends on a local dataset.

## Failure modes

- Duplicate keys silently multiply rows during a join.
- `concat(axis=1)` on frames with different indexes produces NaN gaps instead of side-by-side columns.
- A global preprocessing step learns from the test period.
- Dates remain strings, producing lexical rather than chronological order.
- Invalid numeric text is silently left as object/string data.
- Missing-value statistics are learned before the train/test split.
- `groupby` drops missing keys unless `dropna=False` is chosen deliberately.
- A row window is used where a time window was meant, or a time-window result is aligned positionally with `.values`.
- A daily series built with `groupby(normalize())` silently skips days with no rows.
- Chained assignment appears to succeed but never reaches the parent frame.
- `inplace=True` obscures data flow and does not guarantee lower memory use.
- A printed table has no label identifying the transformation that produced it.

## Interview drill

Given an event table: inspect it, repair one dirty numeric column and one bad timestamp, compute per-user top-2 distinct revenue events, prior-event time gaps, a lagged rolling mean, a trailing 2-day sum joined back on keys, a wide summary flattened from MultiIndex headers, and a validated `pd.merge` with an unmatched-key audit.

Executable reference: [pandas interview refresher](../../notebooks/01-foundations/02_pandas_interview_refresher.ipynb).

## Connections

[[wiki/foundations/numpy|NumPy]] · [[wiki/workflows/sklearn-pipelines|sklearn pipelines]] · [[wiki/workflows/tabular-ml|Tabular ML]] · [[wiki/workflows/sql-interviews|SQL]]

## Sources and provenance

Compiled from the curated pandas recap notebook (`~/Desktop/geek/Coding/old_data_coding/curated_copy/Pandas_Interview_Recap_MiniNotebook.ipynb`, read as an immutable input) and rebuilt on 2026-09-06 from the 2026-09-05 reviewer notes; the per-note decisions are recorded in [[sources/notebook-fundamentals-audit|the notebook fundamentals audit]].
