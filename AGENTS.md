# DataCoding Wiki Schema

## Mission

Maintain a compact, trustworthy, executable knowledge base for ML and data-coding interviews. The primary topics are NumPy, pandas, scikit-learn, PyTorch, visualization, AutoML, SQL for data work, Python OOP, and from-scratch ML algorithms. General LeetCode/DSA material is out of scope unless the user explicitly expands the scope.

## Architecture

This project has three logical layers:

1. **External sources and data (immutable inputs)**
   - Legacy source repository: `$HOME/Desktop/geek/Coding`
   - Dataset/artifact root: `$HOME/Desktop/geek/DataCoding-data`
   - Never modify, delete, or reorganize the legacy repository during normal wiki work.
   - Never copy private documents, transaction exports, environment folders, archives, model weights, logs, notebook checkpoints, or large generated reports into this vault.
2. **Compiled wiki (`wiki/`, `maps/`, `index.md`)**
   - Codex owns the structure and bookkeeping; the user owns intent and corrections.
   - Prefer small, focused, cross-linked Markdown pages over long omnibus notes.
3. **Executable material (`notebooks/`, `src/`, `tests/`, `sql/`)**
   - Reusable logic belongs in `src/datacoding/`; notebooks teach and exercise it.
   - Tests are the source of truth for algorithm correctness.

## Page schema

Every page under `wiki/` and `maps/` must begin with YAML frontmatter:

```yaml
---
type: concept | workflow | map
status: seed | active | mature
tags: [lowercase, tags]
updated: YYYY-MM-DD
---
```

Use these body sections when applicable:

- `# Title`
- one-sentence purpose
- `## Mental model`
- `## Core patterns`
- `## Failure modes`
- `## Interview drill`
- `## Connections`
- `## Sources and provenance`

Do not create empty sections merely to satisfy a template. Claims derived from an external source should record the source path or URL. Human corrections take priority over generated prose and must survive later rewrites.

## Linking rules

- Use Obsidian wikilinks for Markdown knowledge pages: `[[wiki/foundations/numpy|NumPy]]`.
- Use normal relative Markdown links for notebooks, Python modules, tests, and SQL files.
- Every new wiki page must be linked from `index.md` or a map and must link to at least one related page.
- Update renamed links in the same change. Do not leave intentional orphans.

## Notebook standard

Each notebook must start with a Markdown cell containing:

- purpose and expected study time;
- prerequisites;
- whether it is `quick` or `heavy`;
- data/download behavior;
- source provenance.

Code-cell rules:

- Use deterministic random seeds.
- State important shapes next to the relevant code.
- Annotate consequential lines with intent, shape changes, invariants, subtle API behavior, or configuration effects that could be mistaken for data transformations. Prefer a short trailing comment when it stays readable and a preceding comment otherwise; do not narrate obvious syntax.
- Label printed results so every output can be traced to a line or operation. Avoid bare `print(array)` and unexplained final expressions.
- Put reusable implementations in `src/datacoding/`; avoid multiple drifting copies.
- Use train/validation/test separation where evaluation is demonstrated.
- Explain leakage risks and metric choice where relevant.
- Fast, deterministic revision notebooks may retain small text/table outputs.
- Clear outputs from downloads, training-heavy sections, AutoML, large tables, and plots before committing.
- Never embed raw datasets, model weights, caches, secrets, identity documents, or transaction data.

Use [[sources/notebook-fundamentals-audit|the notebook fundamentals audit]] as the compact coverage baseline. Add a concept once at the lowest appropriate layer and prefer a retrieval-worthy pattern over an API catalog.

## Ingest workflow

Process sources one at a time unless the user explicitly requests a batch:

1. Read `index.md`, the relevant map, and related pages first.
2. Inspect the source without modifying it.
3. Decide whether it adds a new concept, strengthens an existing page, contradicts a claim, or is redundant/noisy.
4. Discuss a genuinely consequential ambiguity with the user. Make minor editorial decisions independently.
5. Update the smallest coherent set of wiki pages, executable artifacts, tests, and links.
6. Add or update the source decision in `sources/`.
7. Append a dated entry to `log.md`.
8. Run the relevant tests and `python scripts/check_vault.py`.

## Query workflow

Read `index.md` first, then the smallest relevant pages and executable artifacts. Cite page paths or external sources. If a useful answer creates durable knowledge, file it into the wiki and update the index/log rather than leaving it only in chat.

## Lint workflow

Periodically check for:

- broken or ambiguous links;
- orphaned Markdown pages;
- duplicated concepts or implementations;
- missing provenance;
- stale APIs or executable examples;
- notebook error outputs, huge embedded outputs, or unlabeled prints;
- binaries, datasets, caches, models, or private files accidentally added to the vault;
- contradictions between concept pages and tested code.

Keep the index disposable and reconstructable from the actual pages. Git history, Markdown, code, and tests remain the durable source of truth.
