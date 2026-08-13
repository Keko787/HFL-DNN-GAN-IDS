"""Analytic layout check for the hand-authored SVGs.

The diagrams use one monospace family at known sizes, so every <text>
bounding box is computable without a browser: advance width is ~0.6em per
glyph, plus letter-spacing where the class sets it.

Reports: text clipped by the viewBox, text-vs-text overlaps, and text that
straddles the border of a box it does not belong to.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

sys.path.insert(0, str(Path(__file__).parent))
# Import render_docs rather than diagrams directly: it owns the fully
# populated DIAGRAMS registry (diagrams.py + diagrams_cmp.py), so a new
# diagram is checked the moment it is registered for a document.
import render_docs  # noqa: E402,F401
from diagrams import DIAGRAMS  # noqa: E402

# class -> (font_px, letter_spacing_em)
FONT: Dict[str, Tuple[float, float]] = {
    "": (12.0, 0.0),
    "t-sm": (11.0, 0.0),
    "t-lbl": (10.5, 0.0),
    "t-bad": (10.5, 0.0),
    "t-hd": (10.5, 0.09),
    "t-x": (15.0, 0.0),
}
ADVANCE = 0.60  # monospace advance as a fraction of the em


class Box(NamedTuple):
    x: float
    y: float
    w: float
    h: float
    s: str


def text_boxes(svg: str) -> List[Box]:
    out: List[Box] = []
    for m in re.finditer(
        r'<text x="([-\d.]+)" y="([-\d.]+)"(?: text-anchor="(\w+)")?'
        r'(?: class="([\w -]+)")?>(.*?)</text>', svg):
        x, y = float(m.group(1)), float(m.group(2))
        anchor = m.group(3) or "start"
        cls = (m.group(4) or "").strip()
        # entities occupy one glyph, not their source length
        s = re.sub(r"&(?:[a-zA-Z]+|#x?[0-9a-fA-F]+);", "x", m.group(5))
        size, ls = FONT.get(cls, FONT[""])
        w = len(s) * size * (ADVANCE + ls)
        h = size
        if anchor == "middle":
            x -= w / 2
        elif anchor == "end":
            x -= w
        out.append(Box(x, y - size * 0.78, w, h, s))
    return out


def rect_boxes(svg: str) -> List[Tuple[Box, str]]:
    out: List[Tuple[Box, str]] = []
    for m in re.finditer(
        r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"'
        r'[^>]*class="([\w -]+)"', svg):
        out.append((Box(float(m.group(1)), float(m.group(2)),
                        float(m.group(3)), float(m.group(4)), ""), m.group(5)))
    return out


def overlap(a: Box, b: Box, pad: float = 0.5) -> bool:
    return not (a.x + a.w <= b.x + pad or b.x + b.w <= a.x + pad or
                a.y + a.h <= b.y + pad or b.y + b.h <= a.y + pad)


def viewbox(svg: str) -> Tuple[float, float, float, float]:
    m = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([\d.]+) ([\d.]+)"', svg)
    assert m, "no viewBox"
    return tuple(float(m.group(i)) for i in range(1, 5))  # type: ignore


def check(name: str, svg: str) -> List[str]:
    problems: List[str] = []
    vx, vy, vw, vh = viewbox(svg)
    tb = text_boxes(svg)

    for b in tb:
        if (b.x < vx - 0.5 or b.y < vy - 0.5
                or b.x + b.w > vx + vw + 0.5 or b.y + b.h > vy + vh + 0.5):
            problems.append(
                f"CLIP     {name}: {b.s!r} at ({b.x:.0f},{b.y:.0f}) "
                f"{b.w:.0f}x{b.h:.0f}, viewBox {vw:.0f}x{vh:.0f}")

    for i in range(len(tb)):
        for j in range(i + 1, len(tb)):
            if overlap(tb[i], tb[j]):
                problems.append(
                    f"OVERLAP  {name}: {tb[i].s!r} >< {tb[j].s!r} "
                    f"near ({tb[i].x:.0f},{tb[i].y:.0f})")

    for rb, cls in rect_boxes(svg):
        if "band" in cls or "loopbx" in cls:
            continue
        for b in tb:
            inside = (b.x > rb.x + 1 and b.y > rb.y + 1
                      and b.x + b.w < rb.x + rb.w - 1
                      and b.y + b.h < rb.y + rb.h - 1)
            if not inside and overlap(b, rb, 0.0):
                problems.append(
                    f"STRADDLE {name}: {b.s!r} crosses rect.{cls} "
                    f"at ({rb.x:.0f},{rb.y:.0f}) {rb.w:.0f}x{rb.h:.0f}")
    return problems


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    figs = [
        (fn.__name__, fn)
        for doc in sorted(DIAGRAMS)
        for fn in DIAGRAMS[doc]
    ]
    total = 0
    for name, fn in figs:
        probs = check(name, fn())
        total += len(probs)
        print(f"\n=== {name}: {len(probs)} issue(s) ===")
        for p in probs[:60]:
            print("  " + p)
        if len(probs) > 60:
            print(f"  … {len(probs) - 60} more")
    print(f"\nTOTAL: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
