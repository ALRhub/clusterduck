"""Generate README documentation for ClusterDuckLauncherConf parameters.

Parses hydra_plugins/clusterduck_launcher/config.py, extracts each config
field's name, type, default value and the description comment directly
above its definition, and writes the result into README.md under the
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
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "field":
        return _default_factory_repr(node)
    return ast.unparse(node)


def _description_above(lines: list[str], field_lineno: int) -> str:
    """Collect contiguous comment lines directly above a field definition.

    Lines starting with "##" are section separators and are treated as a
    boundary (not included in the description).
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


def extract_fields() -> list[dict]:
    source = CONFIG_PATH.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)

    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == DATACLASS_NAME
    )

    fields = []
    for node in class_node.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        name = node.target.id
        if name in EXCLUDED_FIELDS:
            continue
        fields.append(
            {
                "name": name,
                "type": ast.unparse(node.annotation),
                "default": _default_repr(node.value),
                "description": _description_above(lines, node.lineno)
                or "No description available.",
            }
        )
    return fields


def render_markdown(fields: list[dict]) -> str:
    entries = []
    for field in fields:
        entry = (
            f"- **{field['name']}** (`{field['type']}`, default: `{field['default']}`)"
        )
        if field["description"] != "No description available.":
            entry += f": {field['description']}"

        entries.append(entry)
    return "\n".join(entries) + "\n"


def update_readme(doc_body: str) -> None:
    content = README_PATH.read_text()
    match = re.search(rf"^{re.escape(HEADING)}\s*$", content, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f'Could not find "{HEADING}" heading in {README_PATH}')

    new_content = content[: match.end()] + "\n" + doc_body
    README_PATH.write_text(new_content)


def main() -> None:
    fields = extract_fields()
    doc_body = render_markdown(fields)
    update_readme(doc_body)


if __name__ == "__main__":
    main()
