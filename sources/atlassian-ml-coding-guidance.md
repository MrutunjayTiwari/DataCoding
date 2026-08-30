---
type: map
status: active
tags: [sources, atlassian, interviews, ml-engineering]
updated: 2026-08-30
---

# Atlassian ML Coding Guidance

This source decision preserves the actionable recruiting guidance without copying the diarized transcript into the vault.

Source: `~/Downloads/Atlassian Durga ML Coding Guidance__whisperx_whisper-large-v3-turbo__diarized.txt` (reviewed as an immutable input on 2026-08-30).

## Retained guidance

| Signal | Decision in the revision material |
|---|---|
| The round is ML-engineering coding, not general data structures and algorithms. | Focus on ML-flavored array/data problems rather than general LeetCode patterns. |
| Coupon recommendation and weighted sampling were named as examples. | Provide executable, tested implementations and scale discussions for both. |
| Code should be documented, modular, understandable, performant, and able to pass all tests. | Separate validation from core logic, label shapes, include edge tests, and state complexity. |
| Candidates should adapt to scale changes and defend choices. | Pair each implementation with baseline/optimized alternatives and large-scale redesign notes. |
| Candidates should clarify the bigger picture, explain system consequences, maintain forward momentum, and unblock themselves. | Include an opening contract, communication checklist, and recovery tactics. |
| AI/code-completion tools are not allowed. | Make the notebook suitable for blank-page recall without such tools. |

## Unconfirmed or excluded

- The candidate asked whether NumPy and pandas familiarity would be sufficient, but the recruiter did not confirm the permitted or expected library set. The notebook therefore uses standard Python + NumPy for the core and labels pandas as optional insurance.
- The call did not define exact production constraints, scoring objectives, test cases, or APIs. Those details are framed as clarifying questions rather than presented as Atlassian requirements.
- ML system design was mentioned as a separate round and is outside this notebook.
- Scheduling discussion, repeated transcript artifacts, and pleasantries add no durable interview knowledge and were excluded.

## Destination

- [Atlassian ML coding revision notebook](../notebooks/07-interview-specific/01_atlassian_notebook.ipynb)
- [[maps/interview-revision|Interview revision map]]

## Connections

[[index|Home]] · [[wiki/foundations/numpy|NumPy]] · [[wiki/foundations/pandas|pandas]] · [[wiki/workflows/tabular-ml|Tabular ML]]
