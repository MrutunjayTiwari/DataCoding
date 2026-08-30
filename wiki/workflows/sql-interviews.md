---
type: workflow
status: active
tags: [sql, interviews, analytics]
updated: 2026-08-30
---

# SQL Interview Patterns

SQL interview fluency is grain control: know what one row represents before and after every join, aggregation, or window.

## Query ritual

1. Restate tables, keys, desired output grain, and SQL dialect.
2. Build and inspect the base join.
3. Aggregate only after confirming join cardinality.
4. Use CTEs to name logical stages.
5. Add windows for row-relative calculations without collapsing rows.
6. State tie, null, and date-boundary behavior.

## Window distinctions

- `ROW_NUMBER`: unique sequence; use with deterministic tie-breakers for deduplication.
- `RANK`: ties share rank and leave gaps.
- `DENSE_RANK`: ties share rank without gaps.
- `LAG`/`LEAD`: compare with prior/next ordered row.
- Aggregate windows retain rows; `GROUP BY` collapses them.
- Explicit `ROWS BETWEEN ...` avoids surprising default frames, especially for `LAST_VALUE` and running calculations.

## Reusable patterns

- Latest row per entity: `ROW_NUMBER() OVER (PARTITION BY id ORDER BY event_time DESC, tie_breaker DESC)` then filter to one.
- Running total: `SUM(value) OVER (PARTITION BY id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`.
- Previous-event gap: `event_time - LAG(event_time) OVER (...)`.
- Conditional aggregation: `SUM(CASE WHEN condition THEN value ELSE 0 END)`.
- Safe ratio: divide by `NULLIF(denominator, 0)`.

## Failure modes

- A one-to-many join inflates sums.
- Ranking order is nondeterministic for ties.
- Filtering in `WHERE` accidentally turns a left join into an inner join.
- `COUNT(column)` ignores nulls while `COUNT(*)` counts rows.
- Inclusive/exclusive date endpoints are unspecified.
- MySQL/PostgreSQL/SQL Server date functions are mixed without naming the dialect.

## Interview drill

Write latest-record deduplication, top-two per group, rolling three-row average, month-over-month change, and a retention-style self-join.

Executable reference: [SQL pattern library](../../sql/interview_patterns.sql).

## Connections

[[wiki/foundations/pandas|pandas equivalents]] · [[wiki/workflows/tabular-ml|Tabular data contracts]] · [[maps/interview-revision|Interview revision]]

## Sources and provenance

Compiled from the legacy SQL refresher PDF, Word window-function refresher, SQL notebook, and interview-strategy note. The broken document-table layout was replaced with searchable Markdown and runnable SQL patterns.
