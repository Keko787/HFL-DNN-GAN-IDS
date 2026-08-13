"""Static audit pass over the HiFINS repo. Read-only; writes a report to stdout."""
from __future__ import annotations

import ast
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# "architecture review" is this tooling — excluded so the audit measures the
# codebase under review and re-runs reproduce the figures quoted in the report.
SKIP_PARTS = {".git", "__pycache__", ".claude", ".idea", ".pytest_cache", ".venv310",
              "architecture review"}

BRANCH_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.ExceptHandler,
    ast.With, ast.AsyncWith, ast.BoolOp, ast.IfExp, ast.comprehension,
    ast.Assert, ast.Match,
)


def iter_py():
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        yield p


def fn_len(node, src_lines):
    end = getattr(node, "end_lineno", None)
    if end is None:
        return 0
    return end - node.lineno + 1


def complexity(node):
    c = 1
    for n in ast.walk(node):
        if isinstance(n, BRANCH_NODES):
            c += 1
        elif isinstance(n, ast.BoolOp):
            c += len(n.values) - 1
    return c


def main():
    long_fns = []
    complex_fns = []
    many_params = []
    broad_excepts = []
    bare_excepts = []
    mutable_defaults = []
    global_stmts = []
    star_imports = []
    prints_in_lib = []
    file_stats = []
    syntax_errors = []
    todo_markers = []
    sleep_calls = []
    thread_spawns = []
    pickle_uses = []
    private_access = []

    for p in iter_py():
        rel = p.relative_to(ROOT).as_posix()
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            continue
        try:
            tree = ast.parse(src, filename=str(p))
        except SyntaxError as e:
            syntax_errors.append((rel, str(e)))
            continue
        lines = src.splitlines()
        nloc = len(lines)
        n_fn = 0
        n_cls = 0

        for i, ln in enumerate(lines, 1):
            s = ln.strip()
            if "TODO" in s or "FIXME" in s or "XXX" in s or "HACK" in s:
                todo_markers.append((rel, i, s[:110]))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_fn += 1
                L = fn_len(node, lines)
                cx = complexity(node)
                a = node.args
                nparams = (
                    len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs)
                    + (1 if a.vararg else 0) + (1 if a.kwarg else 0)
                )
                if L >= 80:
                    long_fns.append((rel, node.lineno, node.name, L))
                if cx >= 20:
                    complex_fns.append((rel, node.lineno, node.name, cx))
                if nparams >= 12:
                    many_params.append((rel, node.lineno, node.name, nparams))
                for d in list(a.defaults) + [d for d in a.kw_defaults if d is not None]:
                    if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                        mutable_defaults.append((rel, node.lineno, node.name))
            elif isinstance(node, ast.ClassDef):
                n_cls += 1
            elif isinstance(node, ast.ExceptHandler):
                body_is_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
                if node.type is None:
                    bare_excepts.append((rel, node.lineno))
                elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
                    if body_is_pass:
                        broad_excepts.append((rel, node.lineno, "except Exception: pass"))
                    else:
                        broad_excepts.append((rel, node.lineno, "except Exception"))
            elif isinstance(node, ast.Global):
                global_stmts.append((rel, node.lineno, ",".join(node.names)))
            elif isinstance(node, ast.ImportFrom) and any(al.name == "*" for al in node.names):
                star_imports.append((rel, node.lineno, node.module or "?"))
            elif isinstance(node, ast.Call):
                f = node.func
                dotted = None
                if isinstance(f, ast.Attribute):
                    base = f.value
                    if isinstance(base, ast.Name):
                        dotted = f"{base.id}.{f.attr}"
                if dotted == "time.sleep":
                    sleep_calls.append((rel, node.lineno))
                if dotted in ("threading.Thread",):
                    thread_spawns.append((rel, node.lineno))
                if dotted in ("pickle.loads", "pickle.dumps", "pickle.load", "pickle.dump"):
                    pickle_uses.append((rel, node.lineno, dotted))
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("_") and not node.attr.startswith("__"):
                    v = node.value
                    if isinstance(v, ast.Attribute) or (isinstance(v, ast.Name) and v.id != "self"):
                        private_access.append((rel, node.lineno, node.attr))

        file_stats.append((rel, nloc, n_fn, n_cls))

    out = {}
    def dump(title, rows, n=40):
        print(f"\n===== {title} ({len(rows)}) =====")
        for r in rows[:n]:
            print("  " + " | ".join(str(x) for x in r))

    print("FILES SCANNED:", len(file_stats))
    dump("SYNTAX ERRORS", syntax_errors)
    dump("FUNCTIONS >= 80 LINES", sorted(long_fns, key=lambda r: -r[3]), 45)
    dump("FUNCTIONS CYCLOMATIC >= 20", sorted(complex_fns, key=lambda r: -r[3]), 35)
    dump("FUNCTIONS WITH >= 12 PARAMS", sorted(many_params, key=lambda r: -r[3]), 35)
    dump("BARE except:", bare_excepts, 30)
    print(f"\n===== BROAD `except Exception` ({len(broad_excepts)}) =====")
    silent = [r for r in broad_excepts if r[2] == "except Exception: pass"]
    print(f"  of which silent (pass): {len(silent)}")
    for r in silent[:30]:
        print("  " + " | ".join(str(x) for x in r))
    dump("STAR IMPORTS", star_imports, 20)
    dump("GLOBAL STATEMENTS", global_stmts, 20)
    dump("MUTABLE DEFAULTS", mutable_defaults, 20)
    dump("PICKLE USE", pickle_uses, 20)
    print(f"\n===== time.sleep call sites: {len(sleep_calls)} =====")
    from collections import Counter
    c = Counter(r[0] for r in sleep_calls)
    for k, v in c.most_common(15):
        print(f"  {v:3d}  {k}")
    print(f"\n===== threading.Thread spawn sites: {len(thread_spawns)} =====")
    c = Counter(r[0] for r in thread_spawns)
    for k, v in c.most_common(15):
        print(f"  {v:3d}  {k}")
    print(f"\n===== TODO/FIXME/HACK markers: {len(todo_markers)} =====")
    c = Counter(r[0] for r in todo_markers)
    for k, v in c.most_common(20):
        print(f"  {v:3d}  {k}")
    print(f"\n===== private attribute access across objects: {len(private_access)} =====")
    c = Counter((r[0], r[2]) for r in private_access)
    for (f, a), v in c.most_common(20):
        print(f"  {v:3d}  {f}  ->  .{a}")

    tot = sum(r[1] for r in file_stats)
    print(f"\nTOTAL LOC (excl. worktrees/pycache): {tot}")


if __name__ == "__main__":
    main()
