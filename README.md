# DataCoding

An Obsidian-friendly, executable wiki for ML and data-coding interview preparation.

Start in [`index.md`](index.md). The wiki holds compact explanations and cross-links; the notebooks hold runnable drills; datasets, models, caches, and generated artifacts live outside the vault at `$HOME/Desktop/geek/DataCoding-data` by default.

## Quick start

```bash
export DATACODING_DATA_DIR="$HOME/Desktop/geek/DataCoding-data"
python3 -m venv "$DATACODING_DATA_DIR/.venv"
source "$DATACODING_DATA_DIR/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -e '.[dev]'
python -m pytest
jupyter lab
```

Install PyTorch in the core environment only when needed:

```bash
python -m pip install -e '.[pytorch]'
```

AutoGluon and PyCaret have different version constraints. Keep them in separate external
environments so neither can downgrade the core stack:

```bash
python3 -m venv "$DATACODING_DATA_DIR/.venv-autogluon"
source "$DATACODING_DATA_DIR/.venv-autogluon/bin/activate"
python -m pip install -e '.[dev,autogluon]'

python3 -m venv "$DATACODING_DATA_DIR/.venv-pycaret"
source "$DATACODING_DATA_DIR/.venv-pycaret/bin/activate"
python -m pip install -e '.[dev,pycaret]'
```

The repository deliberately contains no raw datasets, downloaded model weights, environment folders, or notebook checkpoints.

## Operating model

- `index.md` is the content catalog.
- `log.md` is the append-only activity trail.
- `AGENTS.md` defines how Codex should ingest, update, and lint the wiki.
- `wiki/` contains persistent Markdown knowledge.
- `notebooks/` contains curated, annotated practice.
- `src/datacoding/` contains tested, reusable implementations.
- `sources/` records provenance and migration decisions without copying bulky source files.
