"""Paths for keeping datasets and generated artifacts outside the vault."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "geek" / "DataCoding-data"
DATA_ROOT = Path(os.environ.get("DATACODING_DATA_DIR", DEFAULT_DATA_ROOT)).expanduser()

_AREAS = {"raw", "interim", "processed", "models", "cache", "outputs"}


def external_path(area: str, *parts: str, create_parent: bool = False) -> Path:
    """Return a path below the external data root.

    Parameters
    ----------
    area:
        One of raw, interim, processed, models, cache, or outputs.
    parts:
        Optional path components below the selected area.
    create_parent:
        Create the containing directory when true.
    """

    if area not in _AREAS:
        allowed = ", ".join(sorted(_AREAS))
        raise ValueError(f"Unknown area {area!r}. Expected one of: {allowed}")

    if area == "outputs" and os.environ.get("DATACODING_ARTIFACTS_DIR"):
        base = Path(os.environ["DATACODING_ARTIFACTS_DIR"]).expanduser()
    else:
        base = DATA_ROOT / area

    base = base.resolve()
    result = base.joinpath(*parts).resolve()
    try:
        result.relative_to(base)
    except ValueError as exc:
        raise ValueError("External paths must stay inside their selected data area") from exc

    if create_parent:
        result.parent.mkdir(parents=True, exist_ok=True)
    return result


def ensure_external_layout() -> dict[str, Path]:
    """Create and return the standard external directory layout."""

    paths = {area: external_path(area) for area in sorted(_AREAS)}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
