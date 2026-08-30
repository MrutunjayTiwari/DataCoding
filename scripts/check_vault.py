#!/usr/bin/env python3
"""Deterministic health checks for the Markdown vault and notebooks."""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIRS = {
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "cache",
    "data",
    "datasets",
    "models",
    "outputs",
    "venv",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".csv",
    ".joblib",
    ".log",
    ".parquet",
    ".pkl",
    ".pt",
    ".pth",
    ".zip",
}
PRIVATE_NAME_PATTERNS = ("aadhaar", "adhar", "loantransaction", "mandatory information")
MAX_FILE_BYTES = 2_000_000
MAX_NOTEBOOK_OUTPUT_BYTES = 120_000

WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts and ".obsidian" not in path.parts
    )


def check_file_boundaries(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        lowered_name = path.name.lower()
        if any(part in FORBIDDEN_DIRS or part.endswith(".egg-info") for part in relative.parts):
            errors.append(f"forbidden directory in vault: {relative}")
        if path.is_file():
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append(f"forbidden file type in vault: {relative}")
            if any(pattern in lowered_name for pattern in PRIVATE_NAME_PATTERNS):
                errors.append(f"private-looking filename in vault: {relative}")
            if path.stat().st_size > MAX_FILE_BYTES:
                errors.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative}")


def check_frontmatter(files: list[Path], errors: list[str]) -> None:
    governed_roots = {"maps", "sources", "wiki"}
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts[0] in governed_roots:
            text = path.read_text(encoding="utf-8")
            if not text.startswith("---\n") or "\n---\n" not in text[4:]:
                errors.append(f"missing YAML frontmatter: {relative}")
            for field in ("type:", "status:", "tags:", "updated:"):
                if field not in text.split("---", 2)[1]:
                    errors.append(f"missing frontmatter field {field} in {relative}")


def resolve_wikilink(source: Path, target: str) -> Path | None:
    clean = target.strip().removesuffix(".md")
    direct = ROOT / f"{clean}.md"
    if direct.exists():
        return direct.resolve()
    local = source.parent / f"{clean}.md"
    if local.exists():
        return local.resolve()
    matches = [path for path in markdown_files() if path.stem == Path(clean).name]
    return matches[0].resolve() if len(matches) == 1 else None


def check_links(files: list[Path], errors: list[str]) -> None:
    inbound: dict[Path, int] = defaultdict(int)
    for source in files:
        text = source.read_text(encoding="utf-8")
        for target in WIKILINK.findall(text):
            resolved = resolve_wikilink(source, target)
            if resolved is None:
                errors.append(
                    f"broken or ambiguous wikilink in {source.relative_to(ROOT)}: {target}"
                )
            else:
                inbound[resolved] += 1

        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"broken Markdown link in {source.relative_to(ROOT)}: {raw_target}")
            elif resolved.suffix == ".md":
                inbound[resolved] += 1

    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"maps", "sources", "wiki"} and inbound[path.resolve()] == 0:
            errors.append(f"orphaned knowledge page: {relative}")


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def check_notebooks(errors: list[str]) -> None:
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        relative = path.relative_to(ROOT)
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"invalid notebook JSON {relative}: {exc}")
            continue

        cells = notebook.get("cells", [])
        if not cells or cells[0].get("cell_type") != "markdown":
            errors.append(f"notebook lacks opening Markdown contract: {relative}")
        else:
            opening = source_text(cells[0]).lower()
            for phrase in ("study time", "prerequisites", "mode", "data policy", "provenance"):
                if phrase not in opening:
                    errors.append(f"opening contract missing '{phrase}' in {relative}")

        output_bytes = 0
        for index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            for output in cell.get("outputs", []):
                output_bytes += len(json.dumps(output, ensure_ascii=False))
                if output.get("output_type") == "error":
                    errors.append(
                        f"stored error output in {relative} cell {index}: {output.get('ename', 'error')}"
                    )

            code = source_text(cell)
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                errors.append(f"invalid Python in {relative} cell {index}: {exc}")
                continue

            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    continue
                first_argument = node.args[0] if node.args else None
                is_labeled = (
                    isinstance(first_argument, ast.JoinedStr)
                    or (
                        isinstance(first_argument, ast.Constant)
                        and isinstance(first_argument.value, str)
                    )
                    or (isinstance(first_argument, ast.Name) and first_argument.id == "label")
                )
                if not is_labeled:
                    errors.append(
                        f"possibly unlabeled print in {relative} cell {index}, line {node.lineno}"
                    )

        if output_bytes > MAX_NOTEBOOK_OUTPUT_BYTES:
            errors.append(
                f"notebook outputs exceed {MAX_NOTEBOOK_OUTPUT_BYTES} bytes: {relative} ({output_bytes})"
            )


def main() -> int:
    errors: list[str] = []
    files = markdown_files()
    check_file_boundaries(errors)
    check_frontmatter(files, errors)
    check_links(files, errors)
    check_notebooks(errors)

    if errors:
        print(f"Vault check failed with {len(errors)} issue(s):")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Vault check passed: {len(files)} Markdown files and "
        f"{len(list((ROOT / 'notebooks').rglob('*.ipynb')))} notebooks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
