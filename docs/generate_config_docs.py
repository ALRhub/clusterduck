"""Generate README documentation for ClusterDuckLauncherConf parameters.

Parses hydra_plugins/clusterduck_launcher/config.py, groups fields by the
"##" section comments, and writes the result into README.md under the
"## Reference" heading (replacing any content already there).
"""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "hydra_plugins" / "clusterduck_launcher" / "config.py"
README_PATH = REPO_ROOT / "README.md"

DATACLASS_NAME = "ClusterDuckLauncherConf"
HEADING = "## Reference"

# Fields that are implementation details rather than user-facing parameters.
EXCLUDED_FIELDS = {"_target_"}


def _default_factory_repr(call: ast.Call) -> str:
    for keyword in call.keywords:
        if keyword.arg != "default_factory":
            continue
        factory = keyword.value
        if isinstance(factory, ast.Lambda):
            return ast.unparse(factory.body)
        if isinstance(factory, ast.Name) and factory.id == "dict":
            return "{}"
        if isinstance(factory, ast.Name) and factory.id == "list":
            return "[]"
        return ast.unparse(factory)
    return "field(...)"


def _default_repr(node: ast.expr | None) -> str:
    if node is None:
        return "(required)"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field"
    ):
        return _default_factory_repr(node)
    return ast.unparse(node)


def _description_above(lines: list[str], field_lineno: int) -> str:
    """Collect contiguous comment lines directly above a field definition.

    Lines starting with "##" are section separators treated as a boundary.
    """
    collected: list[str] = []
    i = field_lineno - 2  # index of the line just above the field (0-indexed)
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("##"):
            break
        if stripped.startswith("#"):
            collected.insert(0, stripped.lstrip("#").strip())
            i -= 1
            continue
        break
    return " ".join(collected)


def extract_sections() -> list[dict]:
    """Return a list of sections, each with a title and list of fields."""
    source = CONFIG_PATH.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)

    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == DATACLASS_NAME
    )

    field_by_line = {
        node.lineno: node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id not in EXCLUDED_FIELDS
    }

    class_start = class_node.body[0].lineno
    class_end = class_node.end_lineno
    assert class_end is not None

    sections: list[dict] = []
    current: dict = {"title": None, "fields": []}

    for lineno in range(class_start, class_end + 1):
        stripped = lines[lineno - 1].strip()
        if stripped.startswith("##"):
            sections.append(current)
            current = {"title": stripped.lstrip("#").strip(), "fields": []}
        elif lineno in field_by_line:
            node = field_by_line[lineno]
            assert isinstance(node.target, ast.Name)
            current["fields"].append(
                {
                    "name": node.target.id,
                    "type": ast.unparse(node.annotation),
                    "default": _default_repr(node.value),
                    "description": _description_above(lines, node.lineno),
                }
            )

    sections.append(current)
    return [s for s in sections if s["fields"]]


def render_markdown(sections: list[dict]) -> str:
    parts: list[str] = []
    for section in sections:
        if section["title"]:
            parts.append(f"### {section['title']}\n")
        for f in section["fields"]:
            entry = f"- **{f['name']}** (`{f['type']}`, default: `{f['default']}`)"
            if f["description"]:
                entry += f": {f['description']}"
            parts.append(entry)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def update_readme(doc_body: str) -> None:
    content = README_PATH.read_text()
    match = re.search(rf"^{re.escape(HEADING)}\s*$", content, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f'Could not find "{HEADING}" heading in {README_PATH}')

    new_content = content[: match.end()] + "\n" + doc_body
    README_PATH.write_text(new_content)


def main() -> None:
    sections = extract_sections()
    doc_body = render_markdown(sections)
    update_readme(doc_body)


if __name__ == "__main__":
    main()
