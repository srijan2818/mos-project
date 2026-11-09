"""Every module and every public class / function / method (including nested helpers) is documented."""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_everything_public_has_a_docstring():
    missing = []
    for path in sorted((ROOT / "mos").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py")):
        tree = ast.parse(path.read_text())
        rel = path.relative_to(ROOT)
        if path.stat().st_size and not ast.get_docstring(tree):
            missing.append(f"{rel}: module")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    missing.append(f"{rel}:{node.lineno} {node.name}")
    assert not missing, "undocumented: " + ", ".join(missing)
