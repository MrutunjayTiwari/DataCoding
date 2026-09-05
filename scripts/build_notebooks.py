#!/usr/bin/env python3
"""Build the curated notebook set with deterministic, minimal metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": clean(text)}


def code(text: str, *, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": clean(text),
    }


def contract(
    title: str,
    *,
    study_time: str,
    prerequisites: str,
    mode: str,
    data_policy: str,
    provenance: str,
    goal: str,
) -> dict:
    return md(
        f"""
        # {title}

        {goal}

        - **Study time:** {study_time}
        - **Prerequisites:** {prerequisites}
        - **Mode:** `{mode}`
        - **Data policy:** {data_policy}
        - **Provenance:** {provenance}

        Output convention: every retained textual result begins with a label that identifies the operation that produced it.
        Annotation convention: comments explain intent, shape changes, invariants, subtle API behavior, or configuration side effects; obvious Python syntax is left uncommented.
        """
    )


def write_notebook(relative_path: str, cells: list[dict], *, mode: str) -> None:
    for index, cell in enumerate(cells):
        digest = hashlib.sha1(f"{relative_path}:{index}".encode()).hexdigest()[:8]
        cell["id"] = digest

    notebook = {
        "cells": cells,
        "metadata": {
            "datacoding": {
                "mode": mode,
                "output_policy": (
                    "retain small labeled outputs"
                    if mode == "quick"
                    else "clear outputs after validation"
                ),
            },
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def project_setup_cell() -> dict:
    return code(
        """
        import sys
        from pathlib import Path

        def find_project_root(start=None):
            start = Path.cwd() if start is None else Path(start)
            for candidate in (start, *start.parents):
                if (candidate / "pyproject.toml").exists():
                    return candidate  # Anchor imports to the repository, not the launch directory.
            raise RuntimeError("Run this notebook from inside the DataCoding project")

        PROJECT_ROOT = find_project_root()
        source_dir = str(PROJECT_ROOT / "src")
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)  # Prefer this checkout's reusable implementations.
        """
    )


def build_numpy() -> None:
    cells = [
        contract(
            "NumPy Interview Refresher",
            study_time="40-50 minutes",
            prerequisites="basic Python expressions and loops",
            mode="quick",
            data_policy="no external files or downloads; seeded synthetic arrays only",
            provenance="consolidated from the legacy NumPy refresher variants and advanced curated cells",
            goal="Build fast recall for shapes, combining, broadcasting, indexing, vectorization, ranking, linear algebra, and ML primitives.",
        ),
        code(
            """
            import numpy as np

            np.set_printoptions(precision=3, suppress=True)  # Display only: 3-digit precision and no scientific notation; underlying values are unchanged.
            rng = np.random.default_rng(42)  # Reproducible local generator without global RNG side effects.

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")

            show("Environment | NumPy version", np.__version__)
            """
        ),
        md(
            """
            ## 1. Creation and dtype

            Predict each shape and dtype before running the cell. An ML implementation can silently fail when an integer array truncates a floating-point update.
            """
        ),
        code(
            """
            vector = np.array([1, 2, 3])
            matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
            float_vector = vector.astype(np.float64)  # Promote before operations whose fractional results must survive.
            zeros = np.zeros((2, 3))
            identity = np.eye(3)
            samples = rng.normal(size=(3, 4))

            show("Creation | vector (value, dtype, shape)", (vector, vector.dtype, vector.shape))
            show(
                "Creation | matrix metadata",
                {"shape": matrix.shape, "ndim": matrix.ndim, "size": matrix.size, "dtype": matrix.dtype},
            )
            show("Casting | integer vector to floating dtype", float_vector.dtype)
            show("Creation | zeros (2 x 3)", zeros)
            show("Creation | identity matrix", identity)
            show("Creation | seeded normal samples", samples)
            """
        ),
        md("## 2. Shape changes, views, and copies"),
        code(
            """
            base = np.arange(12)
            reshaped = base.reshape(3, 4)
            raveled = reshaped.ravel()       # View when memory layout permits.
            flattened = reshaped.flatten()  # Independent copy.
            transposed = reshaped.T          # Axis swap; usually a view.
            expanded_front = reshaped[None, :, :]         # Insert batch axis: (1, 3, 4).
            expanded_last = np.expand_dims(reshaped, -1)  # Insert trailing axis: (3, 4, 1).
            squeezed = np.squeeze(expanded_last, axis=-1) # Remove only the named size-1 axis.

            show("Shape | base -> reshaped", f"{base.shape} -> {reshaped.shape}")
            show("Memory | ravel shares memory", np.shares_memory(reshaped, raveled))
            show("Memory | flatten shares memory", np.shares_memory(reshaped, flattened))
            show("Shape/memory | transpose", (transposed.shape, np.shares_memory(reshaped, transposed)))
            show(
                "Shape | add/remove singleton axes",
                (expanded_front.shape, expanded_last.shape, squeezed.shape),
            )
            """
        ),
        md(
            """
            ## 3. Combining and splitting arrays

            `concatenate` joins along an existing axis, so every other dimension must match. `stack` inserts a new axis, so every input shape must match. `vstack` and `hstack` are conveniences; prefer an explicit axis when 1-D behavior could be ambiguous.
            """
        ),
        code(
            """
            left = np.arange(6).reshape(2, 3)
            right = left + 10

            concatenated_rows = np.concatenate([left, right], axis=0)     # Extend existing row axis: (4, 3).
            concatenated_columns = np.concatenate([left, right], axis=1)  # Extend existing column axis: (2, 6).
            stacked_front = np.stack([left, right], axis=0)               # Create new leading axis: (2, 2, 3).
            stacked_last = np.stack([left, right], axis=-1)               # Create new trailing axis: (2, 3, 2).
            equal_halves = np.split(concatenated_rows, 2, axis=0)         # Requires an exact division.
            uneven_chunks = np.array_split(np.arange(7), 3)               # Allows chunk sizes to differ by one.

            show("Combine | input shapes", (left.shape, right.shape))
            show(
                "Combine | concatenate along existing axes",
                {"axis=0": concatenated_rows.shape, "axis=1": concatenated_columns.shape},
            )
            show(
                "Combine | stack along new axes",
                {"axis=0": stacked_front.shape, "axis=-1": stacked_last.shape},
            )
            show(
                "Convenience | vstack/hstack equal explicit concatenate",
                (
                    np.array_equal(np.vstack([left, right]), concatenated_rows),
                    np.array_equal(np.hstack([left, right]), concatenated_columns),
                ),
            )
            show(
                "Split | equal and uneven chunk shapes",
                ([part.shape for part in equal_halves], [part.shape for part in uneven_chunks]),
            )
            """
        ),
        md("## 4. Slicing, boolean masks, and fancy indexing"),
        code(
            """
            A = np.arange(20).reshape(4, 5)
            sliced = A[:2, 1:4]  # Basic slicing preserves a view into A.
            mask = A % 3 == 0
            selected = A[mask]  # Boolean indexing returns a compact 1-D copy.
            rows = np.array([0, 2, 3])
            columns = np.array([1, 4, 0])
            paired = A[rows, columns]  # Pair coordinates elementwise, not as a Cartesian product.
            binary = np.where(A > 12, 1, 0)
            where_rows, where_columns = np.where(A > 12)
            where_coordinates = np.column_stack([where_rows, where_columns])  # Shape: (n_matches, 2).

            show("Indexing | source A", A)
            show("Indexing | basic slice A[:2, 1:4]", sliced)
            show("Memory | slice shares memory with A", np.shares_memory(A, sliced))
            show("Indexing | boolean mask dtype and shape", (mask.dtype, mask.shape))
            show("Indexing | values divisible by three", selected)
            show("Indexing | paired fancy selection A[rows, columns]", paired)
            show("Selection | np.where(A > 12, 1, 0)", binary)
            show("Selection | coordinates returned by np.where(condition)", where_coordinates)
            """
        ),
        md(
            """
            ## 5. Broadcasting

            Align shapes from the right. Adding singleton axes turns pairwise operations into ordinary elementwise arithmetic.
            """
        ),
        code(
            """
            X = rng.normal(size=(5, 2))       # (n=5, d=2)
            centers = rng.normal(size=(3, 2)) # (k=3, d=2)
            mean = X.mean(axis=0, keepdims=True)  # Keep (1, d) for explicit row-wise broadcasting.
            centered = X - mean
            differences = X[:, None, :] - centers[None, :, :]  # All sample-center pairs: (n, k, d).
            squared_distances = np.sum(differences**2, axis=2)  # Reduce features: (n, k).

            show("Broadcast | X, mean, centered shapes", (X.shape, mean.shape, centered.shape))
            show("Broadcast | pairwise difference shape", differences.shape)
            show("Broadcast | pairwise squared distances (n x k)", squared_distances)
            """
        ),
        md("## 6. Reductions and standardization"),
        code(
            """
            X = rng.normal(loc=100, scale=10, size=(6, 3))
            feature_mean = X.mean(axis=0, keepdims=True)  # One statistic per feature.
            feature_std = X.std(axis=0, keepdims=True)
            standardized = (X - feature_mean) / (feature_std + 1e-12)  # Keep constant-feature divisions finite.

            show("Reduction | X.sum(axis=0) shape", X.sum(axis=0).shape)
            show("Reduction | X.sum(axis=1) shape", X.sum(axis=1).shape)
            show("Standardization | feature means after transform", standardized.mean(axis=0))
            show("Standardization | feature std after transform", standardized.std(axis=0))
            """
        ),
        md("## 7. Sorting, top-k, and aligned gather"),
        code(
            """
            scores = rng.normal(size=(3, 8))
            k = 3
            candidate_indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]  # Keep k largest candidates per row; order is arbitrary.
            candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)  # Gather aligned values: (n_rows, k).
            candidate_order = np.argsort(-candidate_scores, axis=1)  # Sort only the k candidates, not every full row.
            topk_indices = np.take_along_axis(candidate_indices, candidate_order, axis=1)  # Restore original column positions.
            topk_scores = np.take_along_axis(scores, topk_indices, axis=1)  # Scores stay aligned with topk_indices.

            show("Top-k | source scores", scores)
            show("Top-k | sorted indices per row", topk_indices)
            show("Top-k | aligned sorted scores", topk_scores)
            """
        ),
        md("## 8. Linear algebra: solve, norms, and SVD"),
        code(
            """
            A = rng.normal(size=(4, 4))
            b = rng.normal(size=4)
            solution = np.linalg.solve(A, b)  # Solve Ax=b directly; avoid forming A^{-1}.
            residual = A @ solution - b       # Numerical correctness check.
            U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)  # Compact factors preserve reconstruction.

            show("Linear algebra | solve residual max abs", np.max(np.abs(residual)))
            show("Linear algebra | row L2 norms", np.linalg.norm(A, axis=1))
            show("Linear algebra | compact SVD shapes", (U.shape, singular_values.shape, Vt.shape))
            """
        ),
        md("## 9. Stable softmax and cosine similarity"),
        code(
            """
            def softmax(logits):
                logits = np.asarray(logits, dtype=float)
                shifted = logits - logits.max(axis=1, keepdims=True)  # Stabilize exp without changing probabilities.
                exponentials = np.exp(shifted)
                return exponentials / exponentials.sum(axis=1, keepdims=True)

            logits = np.array([[1000.0, 1001.0, 999.0], [1.0, 0.0, -1.0]])
            probabilities = softmax(logits)

            left = rng.normal(size=(4, 3))
            right = rng.normal(size=(5, 3))
            left_unit = left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-12)   # Normalize rows; epsilon handles zero vectors.
            right_unit = right / (np.linalg.norm(right, axis=1, keepdims=True) + 1e-12)
            cosine = left_unit @ right_unit.T  # All pairwise cosine similarities: (4, 5).

            show("Softmax | probabilities", probabilities)
            show("Softmax | row sums", probabilities.sum(axis=1))
            show("Cosine similarity | output shape", cosine.shape)
            show("Cosine similarity | first 2 x 3 block", cosine[:2, :3])
            """
        ),
        md("## 10. Scatter-add, sliding windows, and NaN-aware reduction"),
        code(
            """
            repeated_indices = np.array([0, 1, 1, 3, 3, 3])
            values = np.array([10, 1, 1, 5, 2, 2])
            accumulated = np.zeros(5, dtype=int)
            np.add.at(accumulated, repeated_indices, values)  # Accumulate repeated positions instead of overwriting.

            from numpy.lib.stride_tricks import sliding_window_view

            sequence = np.arange(10)
            windows = sliding_window_view(sequence, window_shape=4)  # Overlapping view; avoid writing through it.
            values_with_nan = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

            show("Scatter-add | accumulated repeated indices", accumulated)
            show("Sliding window | shape", windows.shape)
            show("Sliding window | moving averages", windows.mean(axis=1))
            show("NaN-aware reduction | mean vs nanmean", (np.mean(values_with_nan), np.nanmean(values_with_nan)))
            """
        ),
        md(
            """
            ## 11. Retrieval drills

            Re-type these from a blank cell later: concatenate versus stack, standardization, pairwise distances, stable softmax, and vectorized binary metrics.
            """
        ),
        code(
            """
            y_true = rng.integers(0, 2, size=100)
            y_pred = rng.integers(0, 2, size=100)
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            false_positive = np.sum((y_true == 0) & (y_pred == 1))
            false_negative = np.sum((y_true == 1) & (y_pred == 0))
            # Epsilon gives this compact drill a finite zero-denominator policy.
            precision = true_positive / (true_positive + false_positive + 1e-12)
            recall = true_positive / (true_positive + false_negative + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)

            assert np.allclose(probabilities.sum(axis=1), 1.0)
            assert topk_indices.shape == (3, 3)
            assert squared_distances.shape == (5, 3)
            assert concatenated_rows.shape == (4, 3)
            assert stacked_front.shape == (2, 2, 3)
            assert sum(part.size for part in uneven_chunks) == 7
            show("Binary metrics | precision, recall, F1", np.round([precision, recall, f1], 3))
            show("Drill checks | status", "all assertions passed")
            """
        ),
    ]
    write_notebook(
        "notebooks/01-foundations/01_numpy_interview_refresher.ipynb", cells, mode="quick"
    )


def build_pandas() -> None:
    cells = [
        contract(
            "pandas Interview Refresher",
            study_time="40-50 minutes",
            prerequisites="NumPy arrays and basic SQL-style grouping",
            mode="quick",
            data_policy="no external files or downloads; a deterministic event table is created in memory",
            provenance="rebuilt from the curated pandas interview recap notebook",
            goal="Practice selection, missing-data repair, combination, groupby, windows, reshaping, joins, strings, and dates with traceable outputs.",
        ),
        code(
            """
            import numpy as np
            import pandas as pd

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")

            events = pd.DataFrame(
                {
                    "event_id": np.arange(1, 13),
                    "user_id": [101, 101, 102, 101, 103, 102, 103, 103, 101, 102, 103, 102],
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-01 09:00", "2026-01-01 11:00", "2026-01-01 09:30",
                            "2026-01-02 08:00", "2026-01-02 10:15", "2026-01-03 12:00",
                            "2026-01-03 12:30", "2026-01-04 14:00", "2026-01-05 09:00",
                            "2026-01-05 10:00", "2026-01-05 11:00", "2026-01-06 16:00",
                        ]
                    ),
                    "channel": ["web", "app", "web", "store", "app", "web", "store", "app", "web", "store", "web", "app"],
                    "revenue": [20, 35, 15, 60, 10, 45, 55, 25, 80, 30, 50, 70],
                    "note": [f"order_id=ORD-{value:03d}" for value in range(1, 13)],
                }
            )
            show("Source | event table", events.to_string(index=False))
            show("Source | shape and dtypes", (events.shape, events.dtypes.astype(str).to_dict()))
            """
        ),
        md("## 1. `loc`, `iloc`, and safe assignment"),
        code(
            """
            high_value = events.loc[events["revenue"] >= 50, ["event_id", "user_id", "revenue"]]
            high_value_web = events.loc[
                (events["revenue"] >= 50) & (events["channel"] == "web"),  # Parenthesize each mask around & / |.
                ["event_id", "channel", "revenue"],
            ]
            positional = events.iloc[:3, :4]
            labeled = events.copy()  # Make ownership explicit before adding a column.
            labeled.loc[labeled["revenue"] >= 50, "value_band"] = "high"
            labeled.loc[labeled["revenue"] < 50, "value_band"] = "regular"

            show("Selection | loc revenue >= 50", high_value.to_string(index=False))
            show("Selection | parenthesized AND mask", high_value_web.to_string(index=False))
            show("Selection | iloc first 3 rows and 4 columns", positional.to_string(index=False))
            show("Assignment | value_band counts", labeled["value_band"].value_counts().to_string())
            """
        ),
        md(
            """
            ## 2. Missing values, dtype repair, and concatenation

            Audit missingness before choosing a policy. Coerce dirty numeric text with `errors="coerce"`, impute features only from training data, and normally drop rather than impute a missing supervised target. Use `concat` to combine compatible tables by rows or columns; use `merge` when matching keys.
            """
        ),
        code(
            """
            messy = events.copy()
            messy.loc[2, "revenue"] = np.nan
            messy.loc[5, "channel"] = None
            messy["revenue_text"] = messy["revenue"].astype("string")
            messy.loc[4, "revenue_text"] = "unknown"

            numeric_revenue = pd.to_numeric(messy["revenue_text"], errors="coerce")  # Invalid text becomes NaN for audit/repair.
            cleaned = messy.assign(
                revenue=numeric_revenue.fillna(numeric_revenue.median()),  # In ML, learn this fill value on training rows only.
                channel=messy["channel"].fillna("unknown"),
            )
            recombined = pd.concat([events.iloc[:5], events.iloc[5:]], ignore_index=True)  # Stack rows and rebuild a clean index.

            show("Missing data | counts before repair", messy.isna().sum().to_string())
            show("Dtype repair | coerced invalid numeric values", numeric_revenue.head(6).to_string(index=False))
            show("Missing data | counts after selected repairs", cleaned.isna().sum().to_string())
            show("Combine | row-wise concat shape", recombined.shape)
            """
        ),
        md("## 3. Sorting, deduplication, and top-k per group"),
        code(
            """
            latest_per_user = (
                events.sort_values(["user_id", "timestamp", "event_id"])  # event_id resolves timestamp ties deterministically.
                .drop_duplicates("user_id", keep="last")                  # Sorted last row is the latest per user.
                .sort_values("user_id")
            )
            top_two = (
                events.sort_values(["user_id", "revenue"], ascending=[True, False])  # Put each group's winners first.
                .groupby("user_id", group_keys=False)
                .head(2)
            )

            show("Dedup | latest event per user", latest_per_user[["user_id", "event_id", "timestamp"]].to_string(index=False))
            show("Ranking | top 2 revenue events per user", top_two[["user_id", "event_id", "revenue"]].to_string(index=False))
            """
        ),
        md("## 4. `agg` reduces rows; `transform` preserves rows"),
        code(
            """
            user_summary = events.groupby("user_id").agg(  # Collapse to one row per user.
                event_count=("event_id", "size"),
                total_revenue=("revenue", "sum"),
                mean_revenue=("revenue", "mean"),
            )
            with_group_features = events.assign(
                user_mean_revenue=events.groupby("user_id")["revenue"].transform("mean")  # Broadcast one group statistic back to every event.
            )
            with_group_features["above_user_mean"] = (
                with_group_features["revenue"] > with_group_features["user_mean_revenue"]
            )

            show("Groupby agg | one row per user", user_summary.to_string())
            show(
                "Groupby transform | row-aligned feature sample",
                with_group_features[["event_id", "user_id", "revenue", "user_mean_revenue", "above_user_mean"]].head(8).to_string(index=False),
            )
            """
        ),
        md("## 5. Time ordering, shift, gaps, and lagged rolling features"),
        code(
            """
            ordered = events.sort_values(["user_id", "timestamp", "event_id"]).copy()  # Temporal operations require explicit order.
            ordered["previous_timestamp"] = ordered.groupby("user_id")["timestamp"].shift(1)  # Never cross user boundaries.
            ordered["gap_hours"] = (
                ordered["timestamp"] - ordered["previous_timestamp"]
            ).dt.total_seconds() / 3600
            ordered["previous_revenue"] = ordered.groupby("user_id")["revenue"].shift(1)  # Lag before rolling to exclude the current event.
            ordered["prior_two_mean"] = (
                ordered.groupby("user_id")["previous_revenue"]
                .rolling(2, min_periods=1)  # Emit an early value when only one prior event exists.
                .mean()
                .reset_index(level=0, drop=True)  # Remove the group level so values align to ordered's index.
            )
            ordered["month"] = ordered["timestamp"].dt.to_period("M")  # Calendar month period, not a formatted display string.

            show(
                "Time features | previous event, gap, lagged rolling mean",
                ordered[["user_id", "timestamp", "revenue", "gap_hours", "prior_two_mean"]].to_string(index=False),
            )
            """
        ),
        md("## 6. Strings and categorical cleanup"),
        code(
            """
            string_features = events[["event_id", "note", "channel"]].copy()
            string_features["order_id"] = string_features["note"].str.extract(r"(ORD-\\d+)")  # The capture group becomes the new column.
            string_features["channel"] = string_features["channel"].astype("category")  # Store repeated labels as categorical levels.

            show("Strings | extracted order identifiers", string_features.head(6).to_string(index=False))
            show("Categorical | channel categories", string_features["channel"].cat.categories.tolist())
            """
        ),
        md("## 7. Pivot, pivot table, and melt"),
        code(
            """
            revenue_matrix = events.pivot_table(  # Aggregate duplicate user/channel pairs while widening.
                index="user_id",
                columns="channel",
                values="revenue",
                aggfunc="sum",
                fill_value=0,
            )
            revenue_matrix.columns = [f"revenue_{column}" for column in revenue_matrix.columns]
            wide = revenue_matrix.reset_index()
            long = wide.melt(id_vars="user_id", var_name="metric", value_name="value")  # Return metric columns to tidy rows.

            show("Reshape | revenue pivot table", wide.to_string(index=False))
            show("Reshape | melted long form", long.head(9).to_string(index=False))
            """
        ),
        md("## 8. Validated joins and unmatched-key checks"),
        code(
            """
            users = pd.DataFrame(
                {
                    "user_id": [101, 102, 103, 104],
                    "segment": ["growth", "core", "growth", "new"],
                }
            )
            joined = events.merge(
                users,
                on="user_id",
                how="left",
                validate="many_to_one",  # Fail if the supposed lookup table would multiply event rows.
                indicator=True,           # Retain row-level match provenance for the join audit.
            )
            unmatched = joined.loc[joined["_merge"] != "both", ["event_id", "user_id", "_merge"]]

            show("Join | events enriched with segment", joined.head(8).to_string(index=False))
            show("Join audit | unmatched event keys", unmatched.to_string(index=False) if len(unmatched) else "none")
            """
        ),
        md("## 9. Retrieval checks"),
        code(
            """
            assert top_two.groupby("user_id").size().eq(2).all()
            assert joined["event_id"].is_unique
            assert recombined.equals(events)
            assert cleaned[["revenue", "channel"]].notna().all().all()
            assert long.shape[0] == len(wide) * len(revenue_matrix.columns)
            assert ordered.groupby("user_id")["timestamp"].apply(lambda values: values.is_monotonic_increasing).all()

            show("Drill checks | status", "all assertions passed")
            """
        ),
    ]
    write_notebook(
        "notebooks/01-foundations/02_pandas_interview_refresher.ipynb", cells, mode="quick"
    )


def build_oop() -> None:
    cells = [
        contract(
            "Python OOP for ML",
            study_time="30-40 minutes",
            prerequisites="functions, classes, NumPy, and train/test vocabulary",
            mode="quick",
            data_policy="no external files or downloads; seeded synthetic arrays only",
            provenance="new connective material built around the cleaned legacy algorithms",
            goal="Use small estimator classes to separate configuration, fitted state, reusable behavior, and composition.",
        ),
        project_setup_cell(),
        code(
            """
            from dataclasses import dataclass
            import numpy as np

            from datacoding.algorithms import KMeans, KNNClassifier, LinearRegressionGD
            from datacoding.algorithms._validation import NotFittedError

            rng = np.random.default_rng(11)  # Reproducible local generator without global RNG side effects.

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")
            """
        ),
        md(
            """
            ## 1. Configuration versus learned state

            Constructor fields describe choices. Attributes ending in `_` are learned during `fit`.
            """
        ),
        code(
            """
            model = LinearRegressionGD(learning_rate=0.08, max_iter=2_000, l2=0.001)
            configuration = {
                key: value for key, value in vars(model).items() if not key.endswith("_")
            }  # By convention, trailing-underscore attributes are learned state rather than configuration.
            show("Estimator | constructor configuration", configuration)

            X = rng.normal(size=(200, 2))
            y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.7
            model.fit(X, y)  # Populate learned attributes such as coef_ and intercept_.

            learned = {"coef_": model.coef_, "intercept_": model.intercept_, "n_features_in_": model.n_features_in_}
            show("Estimator | learned state after fit", learned)
            """
        ),
        md("## 2. A focused transformer class"),
        code(
            """
            @dataclass  # Generate the constructor for explicit hyperparameter fields only.
            class Standardizer:
                epsilon: float = 1e-12

                def fit(self, X):
                    values = np.asarray(X, dtype=float)
                    self.mean_ = values.mean(axis=0)  # Learned state: one statistic per feature.
                    self.scale_ = values.std(axis=0) + self.epsilon  # Keep constant-feature transforms finite.
                    return self  # Returning self enables estimator-style chaining.

                def transform(self, X):
                    if not hasattr(self, "mean_"):
                        raise NotFittedError("Standardizer must be fitted before transform")
                    return (np.asarray(X, dtype=float) - self.mean_) / self.scale_  # Reuse fitted statistics; never refit here.

                def fit_transform(self, X):
                    return self.fit(X).transform(X)

            standardizer = Standardizer()
            X_standard = standardizer.fit_transform(X)
            show("Transformer | standardized feature means", X_standard.mean(axis=0))
            show("Transformer | standardized feature std", X_standard.std(axis=0))
            """
        ),
        md("## 3. Composition instead of deep inheritance"),
        code(
            """
            @dataclass  # Treat component objects as pipeline configuration.
            class RegressionPipeline:
                transformer: Standardizer
                estimator: LinearRegressionGD

                def fit(self, X, y):
                    transformed = self.transformer.fit_transform(X)  # Learn preprocessing only during pipeline.fit.
                    self.estimator.fit(transformed, y)
                    return self

                def predict(self, X):
                    return self.estimator.predict(self.transformer.transform(X))

            pipeline = RegressionPipeline(
                transformer=Standardizer(),
                estimator=LinearRegressionGD(learning_rate=0.08, max_iter=2_000),
            ).fit(X, y)
            pipeline_mse = np.mean((pipeline.predict(X) - y) ** 2)
            show("Composition | pipeline training MSE sanity check", pipeline_mse)
            """
        ),
        md("## 4. Different estimators, consistent interface"),
        code(
            """
            class_X = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 10], [10, 9]], dtype=float)
            class_y = np.array(["near-zero"] * 3 + ["near-nine"] * 3)
            knn = KNNClassifier(n_neighbors=3).fit(class_X, class_y)

            cluster_X = np.vstack(
                [rng.normal([0, 0], 0.2, size=(40, 2)), rng.normal([4, 4], 0.2, size=(40, 2))]
            )
            kmeans = KMeans(n_clusters=2, random_state=11).fit(cluster_X)

            show("Polymorphic pattern | KNN predictions", knn.predict([[0.2, 0.2], [9.5, 9.1]]))
            show("Polymorphic pattern | K-means learned centers", kmeans.cluster_centers_)
            """
        ),
        md("## 5. Boundary errors should be explicit"),
        code(
            """
            try:
                Standardizer().transform([[1.0, 2.0]])
            except NotFittedError as error:
                show("Expected error | transform before fit", type(error).__name__)

            try:
                model.predict([[1.0, 2.0, 3.0]])
            except ValueError as error:
                show("Expected error | feature-count mismatch", str(error))

            show("OOP drill | status", "configuration, fitted state, composition, and validation demonstrated")
            """
        ),
    ]
    write_notebook("notebooks/01-foundations/03_python_oop_for_ml.ipynb", cells, mode="quick")


def build_linear_models() -> None:
    cells = [
        contract(
            "Linear Models from Scratch",
            study_time="45-60 minutes",
            prerequisites="NumPy broadcasting, gradients, and the estimator contract",
            mode="quick",
            data_policy="no external files or downloads; seeded synthetic train/test splits only",
            provenance="rebuilt from the legacy NumPy ML-from-scratch notebook; rough cells removed and evaluation corrected",
            goal="Exercise linear regression, logistic regression, and perceptron with reusable OOP implementations and held-out checks.",
        ),
        project_setup_cell(),
        code(
            """
            import numpy as np

            from datacoding.algorithms import LinearRegressionGD, LogisticRegressionGD, PerceptronClassifier

            rng = np.random.default_rng(21)  # Reproducible local generator without global RNG side effects.

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")

            def split(X, y, test_fraction=0.25):
                indices = rng.permutation(len(X))  # Apply one permutation to keep X and y aligned.
                cut = int(len(X) * (1 - test_fraction))
                train_index, test_index = indices[:cut], indices[cut:]
                return X[train_index], X[test_index], y[train_index], y[test_index]

            def standardize_train_test(X_train, X_test):
                mean = X_train.mean(axis=0)  # Learn preprocessing from training rows only.
                scale = X_train.std(axis=0) + 1e-12  # Protect constant training features from zero division.
                return (X_train - mean) / scale, (X_test - mean) / scale
            """
        ),
        md("## 1. Linear regression: gradient descent with a least-squares reference"),
        code(
            """
            X = rng.normal(size=(500, 3))
            true_coef = np.array([2.0, -3.0, 0.5])
            y = X @ true_coef + 1.2 + rng.normal(scale=0.3, size=len(X))  # Signal + intercept + irreducible noise.
            X_train, X_test, y_train, y_test = split(X, y)
            X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)

            linear = LinearRegressionGD(learning_rate=0.08, max_iter=3_000, l2=0.0001)
            linear.fit(X_train_scaled, y_train)
            test_prediction = linear.predict(X_test_scaled)
            test_mse = np.mean((test_prediction - y_test) ** 2)

            train_design = np.column_stack([X_train_scaled, np.ones(len(X_train_scaled))])  # Append intercept column.
            test_design = np.column_stack([X_test_scaled, np.ones(len(X_test_scaled))])
            least_squares_parameters = np.linalg.lstsq(train_design, y_train, rcond=None)[0]  # Stable reference solution without an inverse.
            least_squares_prediction = test_design @ least_squares_parameters
            least_squares_mse = np.mean((least_squares_prediction - y_test) ** 2)

            show("Linear regression | train/test shapes", (X_train_scaled.shape, X_test_scaled.shape))
            show("Linear regression | learned coefficients in scaled space", linear.coef_)
            show("Linear regression | held-out MSE", test_mse)
            show("Linear regression | least-squares held-out MSE", least_squares_mse)
            show("Linear regression | first and final objective", (linear.loss_history_[0], linear.loss_history_[-1]))
            """
        ),
        md("## 2. Logistic regression: probability plus threshold"),
        code(
            """
            negative = rng.normal(loc=[-1.5, -1.0], scale=0.9, size=(300, 2))
            positive = rng.normal(loc=[1.5, 1.0], scale=0.9, size=(300, 2))
            X = np.vstack([negative, positive])
            y = np.array([0] * len(negative) + [1] * len(positive))
            X_train, X_test, y_train, y_test = split(X, y)
            X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)

            logistic = LogisticRegressionGD(learning_rate=0.2, max_iter=3_000, l2=0.001)
            logistic.fit(X_train_scaled, y_train)
            probabilities = logistic.predict_proba(X_test_scaled)[:, 1]  # Positive-class probability.
            prediction = logistic.predict(X_test_scaled)
            accuracy = np.mean(prediction == y_test)
            log_loss = -np.mean(
                y_test * np.log(np.clip(probabilities, 1e-12, 1.0))  # Clip only for numerical safety in log.
                + (1 - y_test) * np.log(np.clip(1 - probabilities, 1e-12, 1.0))
            )

            show("Logistic regression | probability range", (probabilities.min(), probabilities.max()))
            show("Logistic regression | held-out accuracy", accuracy)
            show("Logistic regression | held-out log loss", log_loss)
            show("Logistic regression | first five probability/label pairs", np.column_stack([probabilities[:5], y_test[:5]]))
            """
        ),
        md("## 3. Perceptron: update only on mistakes"),
        code(
            """
            negative = rng.normal(loc=[-2.0, -2.0], scale=0.45, size=(120, 2))
            positive = rng.normal(loc=[2.0, 2.0], scale=0.45, size=(120, 2))
            X = np.vstack([negative, positive])
            y = np.array([-1] * len(negative) + [1] * len(positive))  # This perceptron contract uses {-1, +1} labels.
            X_train, X_test, y_train, y_test = split(X, y)

            perceptron = PerceptronClassifier(learning_rate=1.0, max_epochs=50).fit(X_train, y_train)
            prediction = perceptron.predict(X_test)

            show("Perceptron | mistakes per epoch", perceptron.mistakes_per_epoch_)
            show("Perceptron | held-out accuracy", np.mean(prediction == y_test))
            show("Perceptron | learned coefficient/intercept", (perceptron.coef_, perceptron.intercept_))
            """
        ),
        md("## 4. Retrieval checks"),
        code(
            """
            assert test_mse < 0.2
            assert least_squares_mse < 0.2
            assert accuracy > 0.9
            assert np.mean(prediction == y_test) == 1.0
            assert linear.loss_history_[-1] < linear.loss_history_[0]

            show("Algorithm checks | status", "all held-out and convergence assertions passed")
            """
        ),
    ]
    write_notebook(
        "notebooks/02-algorithms-from-scratch/01_linear_models.ipynb", cells, mode="quick"
    )


def build_knn_kmeans() -> None:
    cells = [
        contract(
            "KNN and K-Means from Scratch",
            study_time="40-50 minutes",
            prerequisites="NumPy broadcasting, distances, and estimator state",
            mode="quick",
            data_policy="no external files or downloads; seeded synthetic arrays only",
            provenance="consolidated from the legacy K-means implementation and NumPy distance patterns; KNN added to fill a curriculum gap",
            goal="Contrast supervised neighbor voting with unsupervised centroid updates using the same pairwise-distance primitive.",
        ),
        project_setup_cell(),
        code(
            """
            import numpy as np

            from datacoding.algorithms import KMeans, KNNClassifier

            rng = np.random.default_rng(31)  # Reproducible local generator without global RNG side effects.

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")
            """
        ),
        md("## 1. The shared pairwise-distance primitive"),
        code(
            """
            queries = np.array([[0.0, 0.0], [2.0, 2.0]])
            references = np.array([[0.0, 1.0], [1.0, 0.0], [3.0, 3.0]])
            squared_distances = np.sum(
                (queries[:, None, :] - references[None, :, :]) ** 2,  # Broadcast to (n_queries, n_references, d).
                axis=2,
            )  # Squared distance preserves neighbor ordering without computing square roots.

            show("Distances | query/reference shapes", (queries.shape, references.shape))
            show("Distances | pairwise squared matrix", squared_distances)
            """
        ),
        md("## 2. KNN stores examples and votes at prediction time"),
        code(
            """
            X_train = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 10], [10, 9]], dtype=float)
            y_train = np.array(["low"] * 3 + ["high"] * 3)
            knn = KNNClassifier(n_neighbors=3).fit(X_train, y_train)
            X_query = np.array([[0.2, 0.1], [9.4, 9.2]])

            show("KNN | retained training shape", knn.X_train_.shape)
            show("KNN | neighbor indices", knn._neighbor_indices(X_query))  # Private helper inspected only to expose the voting mechanism.
            show("KNN | predictions", knn.predict(X_query))
            """
        ),
        md(
            """
            ## 3. Feature scale can redefine nearest

            This isolated geometry demo standardizes its full toy matrix. In a supervised workflow, learn mean and standard deviation from the training split only.
            """
        ),
        code(
            """
            scale_demo = np.array([[0.0, 1.0], [1.0, 1000.0], [2.0, 1100.0]])
            mean = scale_demo.mean(axis=0)
            std = scale_demo.std(axis=0) + 1e-12  # Keep a constant feature from producing an infinite scale.
            standardized = (scale_demo - mean) / std  # Put both features on comparable scales.

            raw_distance = np.sum((scale_demo[0] - scale_demo[1:]) ** 2, axis=1)
            standardized_distance = np.sum((standardized[0] - standardized[1:]) ** 2, axis=1)
            show("Scaling | raw squared distances", raw_distance)
            show("Scaling | standardized squared distances", standardized_distance)
            """
        ),
        md("## 4. K-means alternates assignment and update"),
        code(
            """
            X = np.vstack(  # Three compact clouds; labels are intentionally absent.
                [
                    rng.normal([0, 0], 0.25, size=(100, 2)),
                    rng.normal([5, 0], 0.25, size=(100, 2)),
                    rng.normal([2.5, 4], 0.25, size=(100, 2)),
                ]
            )
            kmeans = KMeans(n_clusters=3, max_iter=100, random_state=31).fit(X)

            cluster_counts = np.bincount(kmeans.labels_, minlength=3)  # Keep an explicit zero slot for any empty cluster ID.
            show("K-means | learned centers", kmeans.cluster_centers_)
            show("K-means | cluster counts", cluster_counts)
            show("K-means | inertia", kmeans.inertia_)
            show("K-means | iterations", kmeans.n_iter_)
            """
        ),
        md("## 5. One update written explicitly"),
        code(
            """
            initial_centers = X[[0, 100, 200]].copy()  # One seed per known toy cloud keeps this single update interpretable.
            distances = np.sum((X[:, None, :] - initial_centers[None, :, :]) ** 2, axis=2)  # Assignment costs (n, k).
            labels = np.argmin(distances, axis=1)  # Assign each row to its nearest center.
            updated_centers = np.vstack(
                [X[labels == cluster].mean(axis=0) for cluster in range(3)]
            )  # Replace every center with its assigned-row mean.

            show("One K-means iteration | initial centers", initial_centers)
            show("One K-means iteration | updated centers", updated_centers)
            """
        ),
        md("## 6. Retrieval checks"),
        code(
            """
            assert knn.predict(X_query).tolist() == ["low", "high"]
            assert len(np.unique(kmeans.labels_)) == 3
            assert squared_distances.shape == (2, 3)
            assert np.all(cluster_counts > 0)

            show("Distance-algorithm checks | status", "all assertions passed")
            """
        ),
    ]
    write_notebook("notebooks/02-algorithms-from-scratch/02_knn_kmeans.ipynb", cells, mode="quick")


def build_sklearn() -> None:
    cells = [
        contract(
            "Tabular Pipelines: Classification and Regression",
            study_time="50-65 minutes",
            prerequisites="pandas, train/test splitting, and basic supervised metrics",
            mode="quick",
            data_policy="no external files or downloads; a mixed-type synthetic table is created in memory",
            provenance="consolidated from the curated sklearn pipeline notebook and three legacy pipeline code printouts",
            goal="Build leakage-safe ColumnTransformer pipelines, tune nested parameters, and evaluate held-out classification and regression.",
        ),
        code(
            """
            import numpy as np
            import pandas as pd

            from sklearn.compose import ColumnTransformer
            from sklearn.dummy import DummyClassifier, DummyRegressor
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.metrics import classification_report, f1_score, mean_squared_error, r2_score
            from sklearn.model_selection import RandomizedSearchCV, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

            rng = np.random.default_rng(41)  # Reproducible local generator without global RNG side effects.

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")
            """
        ),
        md("## 1. Build one mixed-type feature table and two targets"),
        code(
            """
            n_rows = 700
            frame = pd.DataFrame(
                {
                    "age": rng.integers(20, 70, size=n_rows).astype(float),
                    "income": rng.normal(70_000, 20_000, size=n_rows),
                    "rooms": rng.integers(1, 7, size=n_rows).astype(float),
                    "region": rng.choice(["north", "south", "east", "west"], size=n_rows),
                    "risk": rng.choice(["low", "medium", "high"], size=n_rows, p=[0.45, 0.35, 0.20]),
                }
            )
            frame.loc[rng.choice(n_rows, 45, replace=False), "income"] = np.nan
            frame.loc[rng.choice(n_rows, 25, replace=False), "region"] = None

            risk_score = frame["risk"].map({"low": 0.0, "medium": 0.8, "high": 1.6})  # Numeric signal used only to synthesize targets.
            region_score = frame["region"].map({"north": 0.3, "south": -0.1, "east": 0.15, "west": 0.0}).fillna(0)
            income_filled = frame["income"].fillna(frame["income"].median())  # Target generation only; model imputation stays inside CV.
            purchase_logit = -4.0 + 0.000045 * income_filled + 0.025 * frame["age"] - 0.9 * risk_score + region_score
            purchase_probability = 1.0 / (1.0 + np.exp(-purchase_logit))
            y_class = (rng.random(n_rows) < purchase_probability).astype(int)
            y_reg = (
                30_000
                + 90 * income_filled
                + 8_000 * frame["rooms"]
                - 12_000 * risk_score
                + 20_000 * region_score
                + rng.normal(0, 8_000, size=n_rows)
            )

            show("Dataset | feature shape", frame.shape)
            show("Dataset | dtypes", frame.dtypes.to_string())
            show("Dataset | missing values", frame.isna().sum().to_string())
            show("Classification target | class fractions", pd.Series(y_class).value_counts(normalize=True).sort_index().to_string())
            """
        ),
        md("## 2. Define preprocessing once"),
        code(
            """
            numeric_features = ["age", "income", "rooms"]
            nominal_features = ["region"]
            ordinal_features = ["risk"]

            numeric_pipeline = Pipeline(  # Fitted independently inside every CV training fold.
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            nominal_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),  # Unseen test categories become all-zero columns.
                ]
            )
            ordinal_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal",
                        OrdinalEncoder(
                            categories=[["low", "medium", "high"]],  # Preserve the domain ordering explicitly.
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,  # Reserve a sentinel outside the known 0..2 category codes.
                        ),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numeric", numeric_pipeline, numeric_features),
                    ("nominal", nominal_pipeline, nominal_features),
                    ("ordinal", ordinal_pipeline, ordinal_features),
                ]
            )
            show("Preprocessing | feature groups", {"numeric": numeric_features, "nominal": nominal_features, "ordinal": ordinal_features})
            """
        ),
        md(
            """
            ## 3. Classification: baseline, pipeline, and nested search

            A dummy model verifies that learned signal beats a trivial policy. This synthetic table is IID, so a stratified random split is appropriate; use grouped or chronological splits when rows share entities or time dependence.
            """
        ),
        code(
            """
            X_train, X_test, y_train, y_test = train_test_split(
                frame,
                y_class,
                test_size=0.25,
                random_state=42,
                stratify=y_class,  # Preserve class proportions across train and held-out test rows.
            )
            # Couple preprocessing and model selection so CV cannot leak fitted transforms.
            classification_pipeline = Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", RandomForestClassifier(random_state=42, n_jobs=1)),
                ]
            )
            classification_search = RandomizedSearchCV(
                classification_pipeline,
                param_distributions={
                    "model__n_estimators": [40, 80, 120],  # model__ addresses the nested pipeline step.
                    "model__max_depth": [None, 5, 10],
                    "model__min_samples_leaf": [1, 3, 6],
                },
                n_iter=4,
                scoring="f1",  # Search the metric used for the final classification comparison.
                cv=3,
                random_state=42,
                n_jobs=1,
            )
            # Search by CV, then refit the winner on all train rows; test stays untouched.
            classification_search.fit(X_train, y_train)
            class_prediction = classification_search.predict(X_test)
            # Minimum useful comparison, not another tuned model.
            classification_baseline = DummyClassifier(strategy="most_frequent").fit(
                X_train, y_train
            )
            baseline_class_prediction = classification_baseline.predict(X_test)
            baseline_f1 = f1_score(y_test, baseline_class_prediction)
            classification_f1 = f1_score(y_test, class_prediction)

            show("Classification | train/test shapes", (X_train.shape, X_test.shape))
            show("Classification | best parameters", classification_search.best_params_)
            show(
                "Classification | baseline versus tuned held-out F1",
                {"dummy_most_frequent": baseline_f1, "random_forest": classification_f1},
            )
            class_report = classification_report(
                y_test,
                class_prediction,
                digits=3,
                zero_division=0,  # Make undefined precision/recall behavior explicit.
            )
            show("Classification | held-out report", class_report)
            """
        ),
        md("## 4. Regression: baseline and pipeline using the same preprocessing contract"),
        code(
            """
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                frame,
                y_reg,
                test_size=0.25,
                random_state=42,
            )
            # Reuse the same preprocessing contract for a different target and model.
            regression_pipeline = Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
                ]
            )
            regression_search = RandomizedSearchCV(
                regression_pipeline,
                param_distributions={
                    "model__n_estimators": [40, 80, 120],
                    "model__max_depth": [None, 6, 12],
                    "model__min_samples_leaf": [1, 3, 6],
                },
                n_iter=4,
                scoring="neg_mean_squared_error",  # sklearn maximizes scores, so loss metrics are negated.
                cv=3,
                random_state=42,
                n_jobs=1,
            )
            # Fit preprocessing per CV fold, then refit the winner on all train rows.
            regression_search.fit(X_train_reg, y_train_reg)
            regression_prediction = regression_search.predict(X_test_reg)
            rmse = np.sqrt(mean_squared_error(y_test_reg, regression_prediction))  # Return error to target units.
            regression_baseline = DummyRegressor(strategy="mean").fit(X_train_reg, y_train_reg)
            baseline_regression_prediction = regression_baseline.predict(X_test_reg)
            baseline_rmse = np.sqrt(
                mean_squared_error(y_test_reg, baseline_regression_prediction)
            )

            show("Regression | best parameters", regression_search.best_params_)
            show(
                "Regression | baseline versus tuned held-out RMSE",
                {"dummy_mean": baseline_rmse, "random_forest": rmse},
            )
            show("Regression | held-out R2", r2_score(y_test_reg, regression_prediction))
            """
        ),
        md("## 5. Inspect the fitted feature space"),
        code(
            """
            fitted_preprocessor = classification_search.best_estimator_.named_steps["preprocess"]  # Inspect the refit winner, not the unfitted template.
            feature_names = fitted_preprocessor.get_feature_names_out()
            transformed_sample = fitted_preprocessor.transform(X_test.head(3))

            show("Pipeline inspection | transformed feature names", feature_names)
            show("Pipeline inspection | transformed sample shape", transformed_sample.shape)
            show("Pipeline inspection | nested parameter prefix example", "model__max_depth")
            """
        ),
        md("## 6. Retrieval checks"),
        code(
            """
            assert len(class_prediction) == len(X_test)
            assert len(regression_prediction) == len(X_test_reg)
            assert transformed_sample.shape[0] == 3
            assert np.isfinite(rmse)
            assert classification_f1 > baseline_f1
            assert rmse < baseline_rmse

            show("Pipeline checks | status", "all shape and held-out-evaluation assertions passed")
            """
        ),
    ]
    write_notebook("notebooks/03-sklearn/01_tabular_pipelines.ipynb", cells, mode="quick")


def build_pytorch_fundamentals() -> None:
    cells = [
        contract(
            "PyTorch Fundamentals and Training Loop",
            study_time="45-60 minutes",
            prerequisites="NumPy shapes, derivatives, and Python classes",
            mode="optional",
            data_policy="no downloads; seeded synthetic tensors only; any checkpoint path resolves outside the vault",
            provenance="consolidated from the legacy PyTorch introduction and training-loop drills",
            goal="Practice tensor construction and reshaping, NumPy interchange, autograd, Dataset/DataLoader, nn.Module, correct train/eval modes, and loss aggregation.",
        ),
        project_setup_cell(),
        code(
            """
            import random
            import numpy as np
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            from datacoding.config import external_path

            def seed_all(seed=51):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")

            seed_all()  # Align Python, NumPy, and PyTorch randomness for this executable example.
            # Select one device reused by model and batches.
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            show("Environment | torch version", torch.__version__)
            show("Environment | selected device", device)
            """
        ),
        md(
            """
            ## 1. Tensor construction, shape operations, and combining

            `torch.cat` joins an existing dimension; `torch.stack` creates a new one. `from_numpy` shares CPU memory with its NumPy input, while `torch.tensor` copies. Before converting a model result to NumPy, use `detach().cpu().numpy()`.
            """
        ),
        code(
            """
            X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            bias = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Shape (4,) aligns with X's trailing feature axis.
            shifted = X + bias
            expanded = X.unsqueeze(0)               # Insert batch dimension: (1, 3, 4).
            permuted = expanded.permute(0, 2, 1)    # Reorder via strides: (1, 4, 3); result may be non-contiguous.
            concatenated = torch.cat([X, X], dim=0) # Extend an existing dimension: (6, 4).
            stacked = torch.stack([X, X], dim=0)    # Create a new dimension: (2, 3, 4).

            numpy_source = np.arange(6, dtype=np.float32).reshape(2, 3)
            shared_tensor = torch.from_numpy(numpy_source)  # Shares CPU storage with numpy_source.
            copied_tensor = torch.tensor(numpy_source)      # Owns independent storage.
            numpy_source[0, 0] = -1.0  # Mutation appears only through the shared tensor.
            detached_numpy = shifted.detach().cpu().numpy()  # Leave autograd, then ensure CPU-backed memory.

            show("Tensor | X shape/dtype/device", (tuple(X.shape), X.dtype, X.device))
            show("Broadcast | X + bias shape", tuple(shifted.shape))
            show("Broadcast | shifted values", shifted)
            show("Shape | unsqueeze then permute", (tuple(expanded.shape), tuple(permuted.shape)))
            show("Combine | cat existing dim versus stack new dim", (tuple(concatenated.shape), tuple(stacked.shape)))
            show(
                "NumPy boundary | from_numpy shares, tensor copies",
                (shared_tensor[0, 0].item(), copied_tensor[0, 0].item()),
            )
            show("NumPy boundary | detached CPU array shape", detached_numpy.shape)
            """
        ),
        md("## 2. Autograd and gradient accumulation"),
        code(
            """
            weight = torch.tensor(2.0, requires_grad=True)
            loss = (weight * 3.0 - 7.0) ** 2
            loss.backward()  # Populate weight.grad through the recorded computation graph.
            first_gradient = weight.grad.item()
            second_loss = (weight * 3.0 - 7.0) ** 2
            second_loss.backward()  # Gradients add to existing .grad by default.
            accumulated_gradient = weight.grad.item()
            weight.grad.zero_()  # Clear in place before a future optimization step.

            show("Autograd | scalar loss", loss.item())
            show("Autograd | first gradient", first_gradient)
            show("Autograd | accumulated after second backward", accumulated_gradient)
            show("Autograd | gradient after zero", weight.grad.item())
            """
        ),
        md("## 3. Dataset and DataLoader contract"),
        code(
            """
            n_rows, n_features = 1_200, 5
            features = torch.randn(n_rows, n_features)
            true_weight = torch.tensor([1.5, -2.0, 0.5, 3.0, -1.0]).reshape(-1, 1)  # Column shape keeps targets (n, 1).
            targets = features @ true_weight + 0.4 + 0.2 * torch.randn(n_rows, 1)

            train_dataset = TensorDataset(features[:900], targets[:900])
            validation_dataset = TensorDataset(features[900:], targets[900:])
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)       # Reshuffle training order each epoch.
            validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)  # Stable evaluation order.
            sample_X, sample_y = next(iter(train_loader))

            show("DataLoader | feature and target batch shapes", (tuple(sample_X.shape), tuple(sample_y.shape)))
            """
        ),
        md("## 4. Model and canonical loops"),
        code(
            """
            class TinyRegressor(nn.Module):
                def __init__(self, n_features):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(n_features, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                    )

                def forward(self, X):
                    return self.network(X)  # Preserve the leading batch dimension: (batch, 1).

            def train_epoch(model, loader, optimizer, loss_fn):
                model.train()  # Enable dropout/batch-norm training behavior; gradient tracking is separate.
                total_loss = 0.0
                total_examples = 0
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad(set_to_none=True)  # Clear prior gradients; None can avoid an eager zero-fill.
                    prediction = model(X_batch)
                    loss = loss_fn(prediction, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(X_batch)  # Convert batch mean to a sample-weighted sum.
                    total_examples += len(X_batch)
                return total_loss / total_examples

            def evaluate(model, loader, loss_fn):
                model.eval()  # Use inference behavior; no_grad below separately disables graph construction.
                total_loss = 0.0
                total_examples = 0
                with torch.no_grad():  # Avoid building graphs that evaluation will never backpropagate through.
                    for X_batch, y_batch in loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        loss = loss_fn(model(X_batch), y_batch)
                        total_loss += loss.item() * len(X_batch)  # Keep the epoch mean correct for a short final batch.
                        total_examples += len(X_batch)
                return total_loss / total_examples
            """
        ),
        code(
            """
            model = TinyRegressor(n_features).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            loss_fn = nn.MSELoss()  # Default batch-mean reduction is reweighted during epoch aggregation.

            history = []
            for epoch in range(1, 7):
                train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
                validation_loss = evaluate(model, validation_loader, loss_fn)
                history.append((train_loss, validation_loss))
                print(f"Training | epoch={epoch:02d} train_mse={train_loss:.4f} validation_mse={validation_loss:.4f}")

            show("Training | first and final loss pairs", (history[0], history[-1]))
            """
        ),
        md("## 5. Save learned state outside the vault"),
        code(
            """
            # Resolve/create an artifact path outside the vault.
            checkpoint_path = external_path("models", "tiny_regressor_state.pt", create_parent=True)
            # Store learned tensors, not the Python model object.
            torch.save(model.state_dict(), checkpoint_path)

            restored = TinyRegressor(n_features).to(device)
            restored.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))  # Restore safely across devices.
            restored_loss = evaluate(restored, validation_loader, loss_fn)

            assert concatenated.shape == (6, 4)
            assert stacked.shape == (2, 3, 4)
            assert accumulated_gradient == 2 * first_gradient
            show("Checkpoint | external path", checkpoint_path)
            show("Checkpoint | restored validation MSE", restored_loss)
            show("Training checks | status", "model modes, no-grad evaluation, and external state_dict verified")
            """
        ),
    ]
    write_notebook("notebooks/04-pytorch/01_pytorch_fundamentals.ipynb", cells, mode="optional")


def build_image_classification() -> None:
    cells = [
        contract(
            "Image Classification Pattern",
            study_time="50-70 minutes",
            prerequisites="PyTorch DataLoader, convolution shapes, and the canonical training loop",
            mode="optional",
            data_policy="no downloads or image files; deterministic circle/square tensors are generated in memory",
            provenance="consolidated from the legacy CIFAR case study and portal dataset-pattern notebooks",
            goal="Follow an image batch from dataset contract through a small CNN, evaluation, and filename-keyed inference.",
        ),
        project_setup_cell(),
        code(
            """
            import random
            import numpy as np
            import pandas as pd
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, Dataset, random_split

            def seed_all(seed=61):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")

            seed_all()  # Align Python, NumPy, and PyTorch randomness for this executable example.
            # Keep model and batches on one selected device.
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            show("Environment | selected device", device)
            """
        ),
        md("## 1. Dataset returns `(C, H, W)`, integer label, and stable identifier"),
        code(
            """
            class ShapeDataset(Dataset):
                classes = ("circle", "square")

                def __init__(self, n_samples=600, image_size=20, seed=61):
                    generator = torch.Generator().manual_seed(seed)  # Dataset-local noise stays reproducible and isolated.
                    self.labels = torch.arange(n_samples) % 2  # Deterministic balanced class sequence.
                    self.filenames = [f"shape_{index:04d}.png" for index in range(n_samples)]
                    coordinates = torch.linspace(-1, 1, image_size)
                    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")  # Pixel-coordinate grids: (H, W).
                    circle = ((xx**2 + yy**2) <= 0.48**2).float()
                    square = ((xx.abs() <= 0.48) & (yy.abs() <= 0.48)).float()
                    templates = torch.stack([circle, square])[:, None, :, :]  # Class templates in (classes, C, H, W).
                    noise = 0.12 * torch.randn(n_samples, 1, image_size, image_size, generator=generator)
                    self.images = (templates[self.labels] + noise).clamp(0, 1)  # Index the template for each label, then perturb pixels.

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, index):
                    return self.images[index], self.labels[index], self.filenames[index]

            dataset = ShapeDataset()
            train_dataset, validation_dataset, test_dataset = random_split(
                dataset,
                [420, 90, 90],
                generator=torch.Generator().manual_seed(61),  # Reproducible, disjoint subsets.
            )
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Shuffle training examples only.
            validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)  # Stable evaluation order.
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            sample_X, sample_y, sample_names = next(iter(train_loader))

            show("Dataset | batch NCHW shape", tuple(sample_X.shape))
            show("Dataset | batch dtype and min/max", (sample_X.dtype, sample_X.min().item(), sample_X.max().item()))
            show("Dataset | first labels and identifiers", (sample_y[:5].tolist(), list(sample_names[:5])))
            """
        ),
        md("## 2. Small CNN produces one logit per class"),
        code(
            """
            class SmallCNN(nn.Module):
                def __init__(self, n_classes=2):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 8, kernel_size=3, padding=1),  # (N, 1, 20, 20) -> (N, 8, 20, 20).
                        nn.ReLU(),
                        nn.MaxPool2d(2),  # Halve spatial dimensions: (N, 8, 10, 10).
                        nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Preserve 10x10, increase channels to 16.
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),  # Collapse any spatial size to one value per channel.
                    )
                    self.classifier = nn.Linear(16, n_classes)

                def forward(self, X):
                    features = self.features(X).flatten(1)  # (N, 16, 1, 1) -> (N, 16); keep batch axis.
                    return self.classifier(features)

            model = SmallCNN().to(device)
            with torch.no_grad():  # Shape probe only; no backward graph is needed.
                sample_logits = model(sample_X.to(device))
            show("Model | input and logits shapes", (tuple(sample_X.shape), tuple(sample_logits.shape)))
            """
        ),
        md("## 3. Train/evaluate without augmenting validation"),
        code(
            """
            def run_epoch(model, loader, loss_fn, optimizer=None):
                training = optimizer is not None  # One loop; optimizer presence selects train versus evaluation.
                model.train(training)             # Switch mode-sensitive layers; gradient context is handled separately.
                total_loss = 0.0
                total_correct = 0
                total_examples = 0
                context = torch.enable_grad() if training else torch.no_grad()  # Build graphs only for updates.
                with context:
                    for X_batch, y_batch, _ in loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        if training:
                            optimizer.zero_grad(set_to_none=True)  # Clear prior batch gradients before backward.
                        logits = model(X_batch)
                        loss = loss_fn(logits, y_batch)
                        if training:
                            loss.backward()
                            optimizer.step()
                        total_loss += loss.item() * len(X_batch)  # Sample-weighted total handles a short final batch.
                        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                        total_examples += len(X_batch)
                return total_loss / total_examples, total_correct / total_examples

            loss_fn = nn.CrossEntropyLoss()  # Expects raw (N, classes) logits and integer class indices.
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(1, 7):
                train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, optimizer)
                validation_loss, validation_accuracy = run_epoch(model, validation_loader, loss_fn)
                message = (
                    f"epoch={epoch:02d} train_loss={train_loss:.4f} "
                    f"train_acc={train_accuracy:.3f} validation_loss={validation_loss:.4f} "
                    f"validation_acc={validation_accuracy:.3f}"
                )
                show("Training | epoch metrics", message)
            """
        ),
        md("## 4. Inference remains keyed by filename"),
        code(
            """
            model.eval()  # Select inference behavior; no_grad below separately disables graph construction.
            rows = []
            with torch.no_grad():
                for X_batch, y_batch, names in test_loader:
                    probabilities = torch.softmax(model(X_batch.to(device)), dim=1).cpu()  # Return probabilities to CPU for reporting.
                    predictions = probabilities.argmax(dim=1)
                    for name, target, prediction, confidence in zip(
                        names,
                        y_batch,
                        predictions,
                        probabilities.max(dim=1).values,
                        strict=True,  # Fail loudly if identifiers, labels, and predictions lose alignment.
                    ):
                        rows.append(
                            {
                                "filename": name,
                                "target": dataset.classes[target.item()],
                                "prediction": dataset.classes[prediction.item()],
                                "confidence": confidence.item(),
                            }
                        )

            inference = pd.DataFrame(rows)
            test_accuracy = (inference["target"] == inference["prediction"]).mean()
            confusion = pd.crosstab(inference["target"], inference["prediction"]).reindex(
                index=dataset.classes,
                columns=dataset.classes,
                fill_value=0,  # Retain classes even if a row/column has no observations.
            )  # Rows are actual classes; columns are predicted classes.
            per_class_recall = pd.Series(
                np.diag(confusion) / confusion.sum(axis=1),  # Correct predictions / actual examples per class.
                index=dataset.classes,
                name="recall",
            )

            assert confusion.to_numpy().sum() == len(inference)
            assert per_class_recall.between(0, 1).all()
            show("Inference | first five filename-keyed predictions", inference.head().to_string(index=False))
            show("Inference | test accuracy", test_accuracy)
            show("Inference | confusion matrix", confusion.to_string())
            show("Inference | per-class recall", per_class_recall.round(3).to_string())
            show(
                "Image checks | status",
                "NCHW, logits, model modes, class-level metrics, and identifier mapping verified",
            )
            """
        ),
    ]
    write_notebook("notebooks/04-pytorch/02_image_classification.ipynb", cells, mode="optional")


def build_visualization() -> None:
    cells = [
        contract(
            "Seaborn Interview EDA",
            study_time="35-45 minutes",
            prerequisites="pandas summaries and basic plotting vocabulary",
            mode="visual",
            data_policy="no downloads; deterministic synthetic customer data only; plot outputs are cleared after validation",
            provenance="rebuilt from the curated Seaborn tutorial around interview questions instead of a plot gallery",
            goal="Match distributions, category comparisons, relationships, and correlation plots to explicit analytical questions.",
        ),
        code(
            """
            import matplotlib
            matplotlib.use("Agg")  # Select a headless rendering backend; this does not change plot data.

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns

            rng = np.random.default_rng(71)  # Reproducible local generator without global RNG side effects.
            sns.set_theme(style="whitegrid", context="notebook")  # Session-wide visual defaults; data are unchanged.

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")

            n_rows = 500
            segment = rng.choice(["new", "core", "premium"], size=n_rows, p=[0.35, 0.45, 0.20])
            tenure = rng.integers(1, 73, size=n_rows)
            base_spend = pd.Series(segment).map({"new": 45, "core": 80, "premium": 145}).to_numpy()  # Segment-level signal.
            spend = base_spend + 0.8 * tenure + rng.normal(0, 25, size=n_rows)
            support_calls = np.maximum(0, rng.poisson(2.5, size=n_rows) - (segment == "premium").astype(int))
            churn_probability = 1 / (1 + np.exp(-(-1.5 - 0.015 * tenure + 0.25 * support_calls)))
            customers = pd.DataFrame(
                {
                    "segment": segment,
                    "tenure_months": tenure,
                    "monthly_spend": spend,
                    "support_calls": support_calls,
                    "churned": rng.random(n_rows) < churn_probability,
                }
            )
            show("Dataset | shape and columns", (customers.shape, customers.columns.tolist()))
            show("Dataset | numeric summary", customers.describe().round(2).to_string())
            """
        ),
        md("## 1. Question: what is the spend distribution?"),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(data=customers, x="monthly_spend", hue="segment", element="step", stat="density", common_norm=False, ax=ax)  # Normalize each segment independently for shape comparison.
            ax.set(title="Monthly spend distribution by segment", xlabel="Monthly spend (currency units)", ylabel="Density")
            fig.tight_layout()
            plt.show()
            plt.close(fig)  # Release the figure after display during repeated notebook runs.
            """
        ),
        md("## 2. Question: how does spend vary by segment?"),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=customers, x="segment", y="monthly_spend", order=["new", "core", "premium"], ax=ax)
            sample = customers.sample(120, random_state=42)  # Show individual observations without saturating the plot.
            sns.stripplot(data=sample, x="segment", y="monthly_spend", order=["new", "core", "premium"], color="black", alpha=0.35, size=3, ax=ax)
            ax.set(title="Spend spread by customer segment", xlabel="Segment", ylabel="Monthly spend")
            fig.tight_layout()
            plt.show()
            plt.close(fig)
            """
        ),
        md("## 3. Question: does tenure relate to spend differently by segment?"),
        code(
            """
            sampled = customers.sample(250, random_state=42)  # Bound visual clutter while keeping sampling reproducible.
            plot = sns.relplot(
                data=sampled,
                x="tenure_months",
                y="monthly_spend",
                hue="segment",
                col="churned",  # Facet the relationship instead of encoding another variable on one crowded axis.
                kind="scatter",
                alpha=0.65,
                height=4,
                aspect=1.0,
            )
            plot.set_axis_labels("Tenure (months)", "Monthly spend")
            plot.figure.suptitle("Tenure and spend, faceted by churn outcome", y=1.04)
            plt.show()
            plt.close(plot.figure)
            """
        ),
        md(
            """
            ## 4. Question: which numeric variables move together?

            Correlation summarizes linear association; inspect distributions and confounding before interpreting it, and never treat it as causal evidence.
            """
        ),
        code(
            """
            correlation = customers[["tenure_months", "monthly_spend", "support_calls", "churned"]].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)  # Diverging scale makes sign and magnitude comparable.
            ax.set_title("Selected numeric correlations")
            fig.tight_layout()
            plt.show()
            plt.close(fig)

            show("Correlation | matrix used by heatmap", correlation.round(3).to_string())
            """
        ),
        md("## 5. Interpretation discipline"),
        code(
            """
            segment_summary = customers.groupby("segment", observed=True).agg(  # observed=True omits unused categorical levels.
                customers=("segment", "size"),
                median_spend=("monthly_spend", "median"),
                churn_rate=("churned", "mean"),
            )
            show("Interpretation | segment summary behind the plots", segment_summary.round(3).to_string())
            show("Visualization checks | status", "each plot has a question, labels, deterministic sampling, and an explicit numerical summary")
            """
        ),
    ]
    write_notebook(
        "notebooks/05-visualization/01_seaborn_interview_eda.ipynb", cells, mode="visual"
    )


def build_automl() -> None:
    cells = [
        contract(
            "AutoML Quickstart: AutoGluon and PyCaret",
            study_time="30 minutes to review; runtime depends on the chosen time budget",
            prerequisites="tabular train/test workflow and metric selection",
            mode="heavy",
            data_policy="no downloads; synthetic tabular data; model artifacts resolve to the external data root; AutoGluon and PyCaret use separate external environments; execution is opt-in with RUN_AUTOML=1",
            provenance="consolidated from the legacy AutoGluon and PyCaret teaching notebooks; errorful exploratory cells removed",
            goal="Use AutoML as a budgeted baseline while preserving a held-out test set and externalizing artifacts.",
        ),
        project_setup_cell(),
        code(
            """
            import importlib.util
            import os

            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split

            from datacoding.config import external_path

            rng = np.random.default_rng(81)  # Reproducible local generator without global RNG side effects.

            def show(label, value):
                print(f"\\n--- {label} ---\\n{value}")

            n_rows = 600
            data = pd.DataFrame(
                {
                    "age": rng.integers(18, 75, size=n_rows),
                    "income": rng.normal(65_000, 18_000, size=n_rows),
                    "channel": rng.choice(["web", "app", "store"], size=n_rows),
                }
            )
            logit = -3.0 + 0.00004 * data["income"] + 0.018 * data["age"] + (data["channel"] == "app") * 0.4  # Controlled synthetic signal.
            probability = 1 / (1 + np.exp(-logit))
            data["label"] = (rng.random(n_rows) < probability).astype(int)
            train_data, test_data = train_test_split(
                data,
                test_size=0.25,
                random_state=42,
                stratify=data["label"],  # Preserve class balance in the untouched test set.
            )

            # Check availability without importing either heavy stack.
            has_autogluon = importlib.util.find_spec("autogluon") is not None
            has_pycaret = importlib.util.find_spec("pycaret") is not None
            run_automl = os.environ.get("RUN_AUTOML") == "1"  # Explicit opt-in prevents accidental heavy runs.

            show("Dataset | train/test shapes", (train_data.shape, test_data.shape))
            show("Optional stack | availability", {"AutoGluon": has_autogluon, "PyCaret": has_pycaret, "RUN_AUTOML": run_automl})
            """
        ),
        md("## 1. AutoGluon with an explicit time budget and external model path"),
        code(
            """
            def run_autogluon(train_frame, test_frame, time_limit=120):
                from autogluon.tabular import TabularPredictor

                # Keep generated artifacts outside the Obsidian vault.
                model_path = external_path("models", "autogluon_tabular_demo")
                predictor = TabularPredictor(
                    label="label",
                    eval_metric="f1",  # Match model selection to the stated classification objective.
                    path=str(model_path),
                )
                predictor.fit(
                    train_data=train_frame,
                    time_limit=time_limit,  # Make compute budget part of the experiment contract.
                    presets="medium_quality",
                )
                held_out_metrics = predictor.evaluate(test_frame)
                # Reporting only: do not reselect a winner from test results.
                held_out_leaderboard = predictor.leaderboard(test_frame)
                return predictor, held_out_metrics, held_out_leaderboard

            if run_automl and has_autogluon:
                autogluon_predictor, autogluon_score, autogluon_leaderboard = run_autogluon(train_data, test_data)
                show("AutoGluon | held-out metrics", autogluon_score)
                show("AutoGluon | leaderboard head", autogluon_leaderboard.head().to_string(index=False))
            else:
                show("AutoGluon | execution status", "skipped; activate the autogluon environment and set RUN_AUTOML=1")
            """
        ),
        md("## 2. PyCaret object-oriented experiment API"),
        code(
            """
            def run_pycaret(train_frame, test_frame):
                from pycaret.classification import ClassificationExperiment

                experiment = ClassificationExperiment()
                experiment.setup(
                    data=train_frame,
                    target="label",
                    session_id=42,  # Reproducible setup and cross-validation.
                    html=False,
                    verbose=False,
                )
                # Select by CV while favoring the quick-baseline model set.
                best_model = experiment.compare_models(turbo=True)
                finalized_model = experiment.finalize_model(best_model)  # Refit the winner on all setup rows.
                predictions = experiment.predict_model(finalized_model, data=test_frame)  # Evaluate once on held-out rows.
                return experiment, finalized_model, predictions

            if run_automl and has_pycaret:
                pycaret_experiment, pycaret_model, pycaret_predictions = run_pycaret(train_data, test_data)
                show("PyCaret | selected model", pycaret_model)
                show("PyCaret | held-out prediction columns", pycaret_predictions.columns.tolist())
            else:
                show("PyCaret | execution status", "skipped; activate the pycaret environment and set RUN_AUTOML=1")
            """
        ),
        md("## 3. Fair-comparison checklist"),
        code(
            """
            checklist = [
                "same train/test definition as the manual baseline",
                "metric selected before model comparison",
                "time and compute budget recorded",
                "test set excluded from selection",
                "leaderboard failures and latency inspected",
                "models and logs stored outside the vault",
            ]
            show("AutoML | review checklist", "\\n".join(f"{index}. {item}" for index, item in enumerate(checklist, start=1)))
            """
        ),
    ]
    write_notebook("notebooks/06-automl/01_automl_quickstart.ipynb", cells, mode="heavy")


def main() -> None:
    build_numpy()
    build_pandas()
    build_oop()
    build_linear_models()
    build_knn_kmeans()
    build_sklearn()
    build_pytorch_fundamentals()
    build_image_classification()
    build_visualization()
    build_automl()
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--fix", str(ROOT / "notebooks")],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(ROOT / "notebooks")],
        check=True,
    )
    print("Built 10 curated notebooks.")


if __name__ == "__main__":
    main()
