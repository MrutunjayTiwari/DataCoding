#!/usr/bin/env python3
"""Execute quick notebooks and statically validate optional/heavy notebooks."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def static_check(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code = "\n".join(
            line
            for line in source_text(cell).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        try:
            ast.parse(code, filename=f"{path}:cell-{index}")
        except SyntaxError as exc:
            raise SyntaxError(f"{path.relative_to(ROOT)} cell {index}: {exc}") from exc


def execute(path: Path, timeout: int) -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError("Install the dev extra before executing notebooks") from exc

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute(cwd=str(ROOT))
    for cell in notebook.get("cells", []):
        cell.get("metadata", {}).pop("execution", None)
    nbformat.write(notebook, path)


def clear_outputs(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.get("metadata", {}).pop("execution", None)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-quick", action="store_true")
    parser.add_argument("--execute-optional", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.chdir(ROOT)
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        mode = notebook.get("metadata", {}).get("datacoding", {}).get("mode", "optional")
        static_check(path)
        should_execute = (mode == "quick" and args.execute_quick) or (
            mode != "quick" and args.execute_optional
        )
        if should_execute:
            execute(path, timeout=args.timeout)
            if mode != "quick":
                clear_outputs(path)
            print(f"executed: {path.relative_to(ROOT)} (mode={mode})")
        else:
            print(f"static check: {path.relative_to(ROOT)} (mode={mode})")


if __name__ == "__main__":
    main()
