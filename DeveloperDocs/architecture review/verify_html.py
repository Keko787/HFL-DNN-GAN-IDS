"""Structural verification of the generated HTML documents.

Checks each file for tag balance, conversion leftovers, required structure,
self-containment, figure accessibility, and id integrity — then checks every
internal and cross-document anchor link across the whole set.
"""
from __future__ import annotations

import re
import sys
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent))
import render_docs  # noqa: E402

REPO = render_docs.REPO

# Derived from the renderer's own source list, so adding a document to
# render_docs.SOURCES automatically brings it under verification.
FILES = [
    out_dir / render_docs.FRONT[src.name]["out"]
    for src, out_dir in render_docs.SOURCES
]

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr", "path", "rect", "line",
        "polyline", "polygon", "circle", "use", "stop"}


class Balance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[tuple] = []
        self.errors: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"line {self.getpos()[0]}: stray </{tag}>")
            return
        top, ln = self.stack.pop()
        if top != tag:
            self.errors.append(
                f"line {self.getpos()[0]}: </{tag}> closes <{top}> opened line {ln}")


def ids_of(text: str) -> Set[str]:
    return set(re.findall(r'\sid="([^"]+)"', text))


def check_one(path: Path, text: str) -> List[str]:
    problems: List[str] = []

    stripped = re.sub(r"<text\b[^>]*>.*?</text>", "", text, flags=re.S)
    b = Balance()
    b.feed(stripped)
    problems += b.errors[:8]
    if b.stack:
        problems.append("unclosed: " + ", ".join(f"<{t}>@{l}" for t, l in b.stack[:8]))

    for pat, why in [
        (r"@@(CODE|DIAGRAM)-\d+@@", "unreplaced placeholder token"),
        (r"```", "raw markdown fence"),
        (r"^\|", "raw markdown table row"),
        (r"\bgraph (TD|LR)\b", "raw mermaid source"),
        (r"sequenceDiagram", "raw mermaid source"),
        (r"<p>#{1,4} ", "unconverted heading"),
        (r"\{#[\w-]+\}", "leftover attr_list anchor syntax"),
        # href only — prose may legitimately name a .md file
        (r'href="[^"]*\.md(?:[#"])', "link still pointing at markdown"),
    ]:
        hits = len(re.findall(pat, text, flags=re.M))
        if hits:
            problems.append(f"{why}: {hits}x  [{pat}]")

    for pat, why in [
        (r"<!doctype html>", "missing doctype"),
        (r'<meta charset="utf-8">', "missing charset"),
        (r'<meta name="viewport"', "missing viewport"),
        (r"prefers-color-scheme:dark", "missing dark-theme block"),
        (r'data-theme="dark"', "missing data-theme override"),
        (r'<nav class="toc"', "missing TOC"),
    ]:
        if not re.search(pat, text, flags=re.I):
            problems.append(f"{why}  [{pat}]")

    for pat, why in [
        (r'src\s*=\s*"https?://', "external script/image"),
        (r'href\s*=\s*"https?://[^"]*\.css', "external stylesheet"),
        (r"@import", "CSS @import"),
        (r"url\(\s*['\"]?https?://", "remote CSS url()"),
    ]:
        if re.search(pat, text, flags=re.I):
            problems.append(f"NOT self-contained: {why}")

    svgs = re.findall(r"<svg\b[^>]*>", text)
    for tag in svgs:
        if 'role="img"' not in tag:
            problems.append("svg without role=img")
        if "aria-label=" not in tag:
            problems.append("svg without aria-label")
    n_fig = len(re.findall(r"<figcaption>", text))
    if n_fig != len(svgs):
        problems.append(f"{len(svgs)} svg but {n_fig} figcaption")

    marker_ids = re.findall(r'<marker id="([^"]+)"', text)
    if len(marker_ids) != len(set(marker_ids)):
        problems.append("duplicate marker ids")
    for ref in set(re.findall(r'marker-end="url\(#([^)]+)\)"', text)):
        if ref not in marker_ids:
            problems.append(f"dangling marker reference #{ref}")

    all_ids = re.findall(r'\sid="([^"]+)"', text)
    dupes = {i for i in all_ids if all_ids.count(i) > 1}
    if dupes:
        problems.append(f"duplicate element ids: {sorted(dupes)[:6]}")

    # code blocks must not have leaked Pygments' own container
    if re.search(r'<div class="highlight">', text):
        problems.append("pygments wrapper leaked into output")

    return problems


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    texts: Dict[Path, str] = {f: f.read_text(encoding="utf-8") for f in FILES}
    id_map: Dict[str, Set[str]] = {f.name: ids_of(t) for f, t in texts.items()}

    total = 0
    for f in FILES:
        probs = check_one(f, texts[f])

        # anchor integrity, same-page and cross-document
        for href in set(re.findall(r'href="([^"]+)"', texts[f])):
            if href.startswith("#"):
                if urllib.parse.unquote(href[1:]) not in id_map[f.name]:
                    probs.append(f"dead same-page anchor {href}")
            elif ".html#" in href:
                target, _, frag = href.partition("#")
                tname = Path(urllib.parse.unquote(target)).name
                if tname in id_map:
                    if urllib.parse.unquote(frag) not in id_map[tname]:
                        probs.append(f"dead cross-doc anchor {href}")
            elif href.endswith(".html"):
                tname = Path(urllib.parse.unquote(href)).name
                if tname not in id_map:
                    probs.append(f"link to unknown html {href}")

        # relative links to repo files should exist on disk
        for href in set(re.findall(r'href="((?:\.\./)+[^"#]+)"', texts[f])):
            p = (f.parent / urllib.parse.unquote(href)).resolve()
            if not p.exists() and p.suffix not in ("", ".html"):
                probs.append(f"relative link misses on disk: {href}")

        total += len(probs)
        size = f.stat().st_size / 1024
        status = "OK" if not probs else f"{len(probs)} ISSUE(S)"
        print(f"\n=== {f.name}  ({size:.0f} KB)  — {status} ===")
        for p in probs[:20]:
            print("   " + p)
        if len(probs) > 20:
            print(f"   … {len(probs) - 20} more")

    print(f"\nTOTAL ISSUES: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
