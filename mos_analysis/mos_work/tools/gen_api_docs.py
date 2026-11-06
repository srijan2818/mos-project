"""Generate docs/api.md from the source code.

Pure `ast`: nothing is imported, so it works without scikit-fem / pyvista / trame.
    python tools/gen_api_docs.py            # rewrite docs/api.md
    python tools/gen_api_docs.py --check    # exit 1 if docs/api.md is out of date
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = ["mos/core/fields.py", "mos/core/material.py", "mos/core/regions.py", "mos/core/loads.py",
           "mos/core/model.py", "mos/core/spec.py", "mos/core/solver.py", "mos/core/post.py",
           "mos/core/meshes.py", "mos/lab/__init__.py", "mos/lab/stress.py", "mos/lab/traction.py",
           "mos/lab/summary.py", "mos/viz.py", "mos/templates.py", "mos/app/main.py"]

HEADER = """# API reference

Generated from the source by `python tools/gen_api_docs.py`; do not edit by hand
(`tests/test_docs.py` fails if this file is out of date).
Only public names are listed. Conventions used throughout: SI units, arrays with the component
axis first, e.g. positions `(3, ...)`, tensors `(3, 3, ...)`. See [theory.md](theory.md).
"""


def _sig(fn: ast.FunctionDef) -> str:
    args = re.sub(r"^(self|cls)(, |$)", "", ast.unparse(fn.args))
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    return f"{fn.name}({args}){ret}"


def _doc(node) -> str:
    """Docstring as markdown; text with indentation/alignment goes in a fence so it stays readable."""
    d = ast.get_docstring(node)
    if not d:
        return "*(undocumented)*"
    d = inspect.cleandoc(d)
    if any(l.startswith(" ") or "   " in l for l in d.splitlines()):
        return "```text\n" + d + "\n```"
    return d


def _decos(node) -> str:
    names = [ast.unparse(d) for d in node.decorator_list]
    return f"  _({', '.join('@' + n for n in names)})_" if names else ""


def _public(n) -> bool:
    return not n.name.startswith("_")


def _module_section(rel: str) -> str:
    path = ROOT / rel
    tree = ast.parse(path.read_text())
    dotted = rel[:-3].replace("/", ".")
    out = [f"\n## `{dotted}`\n", _doc(tree) + "\n"]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _public(node):
            out += [f"### `{_sig(node)}`{_decos(node)}\n", _doc(node) + "\n"]
        elif isinstance(node, ast.ClassDef) and _public(node):
            out += [f"### class `{node.name}`{_decos(node)}\n", _doc(node) + "\n"]
            fields = [n for n in node.body if isinstance(n, ast.AnnAssign)]
            if fields:
                out.append("Fields: " + ", ".join(f"`{ast.unparse(f)}`" for f in fields) + "\n")
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and _public(m):
                    out += [f"#### `{node.name}.{_sig(m)}`{_decos(m)}\n", _doc(m) + "\n"]
        elif isinstance(node, ast.Assign) and rel.endswith("templates.py"):
            out.append("`TEMPLATES` maps each menu name to a model description: "
                       + ", ".join(f"*{k.value}*" for k in node.value.keys) + ".\n")
    return "\n".join(out)


def generate() -> str:
    """The full text of docs/api.md for the current source."""
    return HEADER + "".join(_module_section(m) for m in MODULES)


if __name__ == "__main__":
    target = ROOT / "docs" / "api.md"
    text = generate()
    if "--check" in sys.argv:
        sys.exit(0 if target.exists() and target.read_text() == text else 1)
    target.write_text(text)
    print(f"wrote {target} ({len(text.splitlines())} lines)")
