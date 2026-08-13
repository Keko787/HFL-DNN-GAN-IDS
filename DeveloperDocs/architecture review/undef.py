"""Find names referenced in a module that are never bound (imports, defs, assigns, builtins).

Approximation: collects every binding at any scope in the file (module + nested),
then reports loads not covered. Over-approximates bindings => low false positives.
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {".git", "__pycache__", ".claude", ".idea", ".pytest_cache", ".venv310",
        "architecture review"}
BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__", "__builtins__", "self", "cls"}


class Collector(ast.NodeVisitor):
    def __init__(self):
        self.bound = set()
        self.loads = []  # (name, lineno)

    def visit_Import(self, node):
        for a in node.names:
            self.bound.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.bound.add(a.asname or a.name)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)
        self._args(node.args)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._args(node.args)
        self.generic_visit(node)

    def _args(self, a):
        for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
            self.bound.add(x.arg)
        if a.vararg:
            self.bound.add(a.vararg.arg)
        if a.kwarg:
            self.bound.add(a.kwarg.arg)

    def visit_ClassDef(self, node):
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.bound.update(node.names)

    def visit_Nonlocal(self, node):
        self.bound.update(node.names)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)
        else:
            self.loads.append((node.id, node.lineno))

    def visit_alias(self, node):
        pass


def strings_in_annotations(tree):
    """Names appearing inside annotations (which may be strings under PEP 563)."""
    out = []
    for node in ast.walk(tree):
        ann = getattr(node, "annotation", None)
        rets = getattr(node, "returns", None)
        for a in (ann, rets):
            if a is None:
                continue
            for n in ast.walk(a):
                if isinstance(n, ast.Name):
                    out.append((n.id, n.lineno))
                elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                    try:
                        sub = ast.parse(n.value, mode="eval")
                    except SyntaxError:
                        continue
                    for m in ast.walk(sub):
                        if isinstance(m, ast.Name):
                            out.append((m.id, n.lineno))
    return out


hits = []
for p in ROOT.rglob("*.py"):
    if any(part in SKIP for part in p.parts):
        continue
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception:
        continue
    c = Collector()
    c.visit(tree)
    ann = strings_in_annotations(tree)
    unresolved = {}
    for name, ln in c.loads + ann:
        if name in c.bound or name in BUILTINS:
            continue
        unresolved.setdefault(name, ln)
    if unresolved:
        rel = p.relative_to(ROOT).as_posix()
        for name, ln in sorted(unresolved.items(), key=lambda kv: kv[1]):
            hits.append((rel, ln, name))

print(f"UNRESOLVED NAME REFERENCES: {len(hits)}")
for rel, ln, name in hits:
    print(f"  {rel}:{ln}  {name}")
