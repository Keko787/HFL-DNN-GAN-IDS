"""Build the full FeRRy paper, DeveloperDocs/paper/ferry_paper.tex.

The paper is front.tex (preamble through related works), then the sections of
methods.tex, setup.tex (experimental setup and design), evaluation.tex (results
and discussion) and back.tex (the text between their
"%%%%% BEGIN SECTION %%%%%" and "%%%%% END SECTION %%%%%" markers). Each
"%%TABLE:<label>%%" or "%%FIGURE:<label>%%" line in a section is replaced by the
float of that label from results/exp5/paper/exp5_tables.tex or exp5_figures.tex,
which scripts/exp5/paper_tables.py and paper_figures.py generate from the scores.
The figure PDFs those floats include (Figures/exp5/...) are copied into
DeveloperDocs/paper/Figures/exp5/, so the folder compiles as it is.

The script then checks the result: every placeholder filled, every \\ref and
\\eqref has a \\label, every \\cite key is in biblio.bib, and environments are
balanced. It exits non-zero on a failed check.

    py -3.11 DeveloperDocs/paper/assemble.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENERATED = ROOT / "results" / "exp5" / "paper"
OUT = HERE / "ferry_paper.tex"
BIB = HERE / "biblio.bib"
FIGURES = HERE / "Figures"

BEGIN = "%%%%% BEGIN SECTION %%%%%"
END = "%%%%% END SECTION %%%%%"
FLOAT = re.compile(r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}", re.S)
PLACEHOLDER = re.compile(r"^%%(TABLE|FIGURE):([^%]+)%%\s*$", re.M)


def section(path: Path) -> str:
    """The lines between the marker lines (a marker quoted inside a line is text)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    marks = [line.strip() for line in lines]
    if marks.count(BEGIN) != 1 or marks.count(END) != 1:
        sys.exit(f"{path.name}: expected one BEGIN and one END marker line")
    return "\n".join(lines[marks.index(BEGIN) + 1:marks.index(END)]).strip("\n")


def floats(path: Path) -> dict[str, str]:
    """The floats of a generated file, by label."""
    out = {}
    for m in FLOAT.finditer(path.read_text(encoding="utf-8")):
        label = re.search(r"\\label\{([^}]+)\}", m.group(0))
        if label:
            out[label.group(1)] = m.group(0)
    return out


def fill(text: str, blocks: dict[str, str], used: list[str]) -> str:
    def swap(m: re.Match) -> str:
        label = m.group(2).strip()
        if label not in blocks:
            sys.exit(f"no generated float labelled {label}")
        used.append(label)
        return blocks[label]
    return PLACEHOLDER.sub(swap, text)


def uncommented(text: str) -> str:
    """The text with LaTeX comments removed (an escaped \\% is kept)."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def check(tex: str) -> list[str]:
    problems = []
    body = uncommented(tex)
    left = re.findall(r"%%[A-Z-]+[:%]", tex)
    if left:
        problems.append(f"unfilled placeholders: {left}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for ref in sorted(set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", body)) - labels):
        problems.append(f"\\ref to a missing label: {ref}")
    bib = BIB.read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,\s]+),", bib, re.M))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body) for k in group.split(",")}
    for key in sorted(cited - keys):
        problems.append(f"\\cite key not in biblio.bib: {key}")
    # biblio.bib holds several papers twice under different keys; citing both
    # would list the paper twice.
    by_title: dict[str, list[str]] = {}
    for key in sorted(cited & keys):
        entry = bib[bib.index("{" + key + ","):]
        title = re.search(r"\btitle\s*=\s*(.+)", entry)
        norm = re.sub(r"[^a-z]", "", title.group(1).lower()) if title else key
        by_title.setdefault(norm, []).append(key)
    for same in by_title.values():
        if len(same) > 1:
            problems.append(f"one paper cited under several keys: {same}")
    begins = re.findall(r"\\begin\{([^}]+)\}", body)
    ends = re.findall(r"\\end\{([^}]+)\}", body)
    for env in sorted(set(begins) | set(ends)):
        if begins.count(env) != ends.count(env):
            problems.append(f"unbalanced environment {env}: {begins.count(env)} begin, {ends.count(env)} end")
    if body.count("{") - body.count("\\{") != body.count("}") - body.count("\\}"):
        problems.append("unbalanced braces")
    for n, line in enumerate(body.splitlines(), 1):
        if re.search(r"\bHERMES\b", line) and "includegraphics" not in line:
            problems.append(f"line {n} still names HERMES")
    return problems


def main() -> None:
    blocks = floats(GENERATED / "exp5_tables.tex") | floats(GENERATED / "exp5_figures.tex")
    used: list[str] = []
    parts = [
        (HERE / "front.tex").read_text(encoding="utf-8").rstrip("\n"),
        section(HERE / "methods.tex"),
        fill(section(HERE / "setup.tex"), blocks, used),
        fill(section(HERE / "evaluation.tex"), blocks, used),
        section(HERE / "back.tex"),
        r"\end{document}",
    ]
    header = ("% ferry_paper.tex: GENERATED by DeveloperDocs/paper/assemble.py from front.tex,\n"
              "% methods.tex, setup.tex, evaluation.tex, back.tex and results/exp5/paper/*.tex.\n"
              "% Edit those sources and rerun the script; edits here are overwritten.\n")
    tex = header + "\n\n".join(parts) + "\n"

    # The generated figures sit under Figures/exp5/ in the paper (paper_figures.py
    # writes those paths); results/exp5/paper/figures/ holds them flat.
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{Figures/([^}]+)\}", uncommented(tex)):
        src = GENERATED / "figures" / Path(name).name
        dest = FIGURES / name
        if name.startswith("exp5/") and src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        elif not dest.exists():
            print(f"note: Figures/{name} is not in the repo; add it before compiling")

    OUT.write_text(tex, encoding="utf-8")
    words = len(re.findall(r"[A-Za-z]{2,}", uncommented(tex.split(r"\begin{document}", 1)[1])))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(tex.splitlines())} lines, about {words} words")
    print("generated floats:", ", ".join(used))

    problems = check(tex)
    for p in problems:
        print("CHECK:", p)
    if problems:
        sys.exit(1)
    print("checks passed")


if __name__ == "__main__":
    main()
