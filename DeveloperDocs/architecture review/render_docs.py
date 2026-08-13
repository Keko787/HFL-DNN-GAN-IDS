"""Render the HiFINS review markdown into house-style HTML.

Self-contained output: one file per document, embedded CSS, hand-authored
inline SVG for the diagrams, Pygments tokenisation with theme-mapped colours.
No external requests, no runtime dependencies. Re-runnable — regenerate after
any edit to the markdown source.

    python render_docs.py
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound

sys.path.insert(0, str(Path(__file__).parent))
from diagrams import DIAGRAMS  # noqa: E402
from diagrams_cmp import d7_policy_authority, d8_ablation  # noqa: E402

DIAGRAMS["CEDA_vs_HERMES.md"] = [d7_policy_authority, d8_ablation]

# DeveloperDocs/architecture review/render_docs.py -> repo root
REPO = Path(__file__).resolve().parents[2]
ARCH = REPO / "DeveloperDocs" / "architecture documents"
REVIEW = REPO / "DeveloperDocs" / "Codebase Review"
COMPARE = REPO / "DeveloperDocs" / "Comparative Analysis"
HOUSE_CSS = Path(__file__).parent / "house.css.html"

# --------------------------------------------------------------------------- #
# per-document front matter
# --------------------------------------------------------------------------- #

FRONT: Dict[str, Dict[str, str]] = {
    "System_Architecture_Overview.md": {
        "out": "System_Architecture_Overview.html",
        "eyebrow": "Architecture review · 2026-08-10 · main @ 35e2eeb",
        "eyebrow2": "Whole repository",
        "eyebrow3": "Review only — no code changed",
        "title": "HiFINS — System Architecture, As&nbsp;Built",
        "deck": ("What the code <b>is</b>, not what the design docs say it should be. "
                 "Two independent systems in one repository, joined by a bridge that "
                 "does not carry traffic — and everything that follows from that."),
    },
    "HERMES_Architecture.md": {
        "out": "HERMES_Architecture.html",
        "eyebrow": "Architecture review · subsystem deep dive",
        "eyebrow2": "hermes/ · 69 modules · 14,147 LOC",
        "eyebrow3": "Review only — no code changed",
        "title": "HERMES — Subsystem Architecture",
        "deck": ("The scheduling and transport substrate: seven cooperating programs "
                 "across four tiers, framework-free by design. <b>The strongest code "
                 "in the repository</b> — and the three places that discipline "
                 "stopped."),
    },
    "HIFINS_Architecture.md": {
        "out": "HIFINS_Architecture.html",
        "eyebrow": "Architecture review · subsystem deep dive",
        "eyebrow2": "App · Config · experiments · 46.8K LOC",
        "eyebrow3": "Review only — no code changed",
        "title": "HIFINS — ML / Training Stack Architecture",
        "deck": ("The half of the system that can actually train a model, and the "
                 "half nobody can safely change. <b>Configuration-as-code</b> and "
                 "positional-tuple plumbing, and everything downstream of them."),
    },
    "00_Critical_Problem_Areas.md": {
        "out": "00_Critical_Problem_Areas.html",
        "eyebrow": "Codebase review · findings register",
        "eyebrow2": "27 findings · P0 / P1 / P2",
        "eyebrow3": "Review only — no code changed",
        "title": "Critical Problem Areas",
        "deck": ("Every finding verified against source, with a file:line citation "
                 "and the specific mechanism of failure — <b>not a code smell, a "
                 "consequence</b>. Five of them are P0."),
    },
    "01_Refactoring_Strategy.md": {
        "out": "01_Refactoring_Strategy.html",
        "eyebrow": "Codebase review · proposal",
        "eyebrow2": "26 items · 7 phases · ~13 weeks",
        "eyebrow3": "Requires approval before any code change",
        "title": "Refactoring Strategy",
        "deck": ("Sequenced by what each step unblocks, not by severity. "
                 "Behaviour-preserving by construction except for <b>three items "
                 "quarantined into their own approval-gated phase</b>."),
    },
    "HERMES_Findings_and_Refactoring.md": {
        "out": "HERMES_Findings_and_Refactoring.html",
        "eyebrow": "Codebase review · subsystem",
        "eyebrow2": "hermes/ · 10 findings",
        "eyebrow3": "Review only — no code changed",
        "title": "HERMES — Findings and Refactoring Plan",
        "deck": ("The reference standard for the rest of the repository. Its defects "
                 "fall into three clean groups — <b>and none of them needs a "
                 "rewrite</b>."),
    },
    "HERMES_Production_Code.md": {
        "out": "HERMES_Production_Code.html",
        "eyebrow": "Codebase review · reference implementation",
        "eyebrow2": "hermes/ · 7 defect classes",
        "eyebrow3": "Proposal — not applied",
        "title": "HERMES — Production-Grade Code",
        "deck": ("Every change below is <b>behaviour-preserving in the currently "
                 "tested regime</b>. Where one alters behaviour outside it — the "
                 ">30 s idle case, which is broken today — that is stated "
                 "explicitly."),
    },
    "HIFINS_Findings_and_Refactoring.md": {
        "out": "HIFINS_Findings_and_Refactoring.html",
        "eyebrow": "Codebase review · subsystem",
        "eyebrow2": "App · Config · 9 findings",
        "eyebrow3": "Review only — no code changed",
        "title": "HIFINS — Findings and Refactoring Plan",
        "deck": ("21,278 LOC of training loops with <b>zero test coverage</b>, and "
                 "21 CLI hyperparameters that nothing reads. One pattern choice "
                 "causes almost all of it."),
    },
    "CEDA_vs_HERMES.md": {
        "out": "CEDA_vs_HERMES.html",
        "eyebrow": "Comparative analysis · 2026-08-10",
        "eyebrow2": "MS thesis defence · 37 slides",
        "eyebrow3": "Analysis only — nothing modified",
        "title": "CEDA vs HERMES",
        "deck": ("Two systems from one lab, solving structurally the same scheduling "
                 "problem — and making <b>opposite</b> choices about how much the "
                 "learned policy is allowed to decide. Plus the untracked prototype "
                 "sitting between them."),
    },
    "HIFINS_Production_Code.md": {
        "out": "HIFINS_Production_Code.html",
        "eyebrow": "Codebase review · reference implementation",
        "eyebrow2": "App · Config · net −24,700 LOC",
        "eyebrow3": "Proposal — not applied",
        "title": "HIFINS — Production-Grade Code",
        "deck": ("Packaging, typed configuration, registry factories, and the "
                 "<b>adapter package that finally joins the two stacks</b> — with "
                 "the two behaviour-changing items flagged in place."),
    },
}

SOURCES: List[Tuple[Path, Path]] = [
    (ARCH / "System_Architecture_Overview.md", ARCH),
    (ARCH / "Hermes" / "HERMES_Architecture.md", ARCH / "Hermes"),
    (ARCH / "HIFins" / "HIFINS_Architecture.md", ARCH / "HIFins"),
    (REVIEW / "00_Critical_Problem_Areas.md", REVIEW),
    (REVIEW / "01_Refactoring_Strategy.md", REVIEW),
    (REVIEW / "Hermes" / "HERMES_Findings_and_Refactoring.md", REVIEW / "Hermes"),
    (REVIEW / "Hermes" / "HERMES_Production_Code.md", REVIEW / "Hermes"),
    (REVIEW / "HIFINS" / "HIFINS_Findings_and_Refactoring.md", REVIEW / "HIFINS"),
    (REVIEW / "HIFINS" / "HIFINS_Production_Code.md", REVIEW / "HIFINS"),
    (COMPARE / "CEDA_vs_HERMES.md", COMPARE),
]

# every rendered doc, so cross-links between them stay live
LINK_REWRITES = {name: front["out"] for name, front in FRONT.items()}


# --------------------------------------------------------------------------- #
# CSS additions
# --------------------------------------------------------------------------- #

EXTRA_CSS = """
  /* ---------- Additions for the generated review docs ---------- */
  main h2 { max-width: 32ch }
  main table { min-width: 560px }
  .tablewrap table thead th { position: static }

  /* finding / item sub-headings */
  main h3{
    scroll-margin-top:18px; display:flex; align-items:baseline;
    gap:9px; flex-wrap:wrap;
  }
  main h3 .fid{
    font-family:var(--font-mono); font-size:12.5px; font-weight:700;
    color:var(--accent); letter-spacing:.02em; flex:0 0 auto;
  }
  :root[data-theme="dark"] main h3 .fid{ color:var(--accent-2) }
  @media(prefers-color-scheme:dark){ main h3 .fid{ color:var(--accent-2) } }
  p.pillrow{ display:flex; gap:8px; flex-wrap:wrap; margin:-.35em 0 1.1em }
  .anchor-alias{ display:block; height:0; scroll-margin-top:18px }

  /* code blocks */
  .codewrap{ position:relative; margin:1.15em 0 1.4em; max-width:var(--prose) }
  .codewrap.wide{ max-width:none }
  pre.code{
    font-family:var(--font-mono); font-size:12.5px; line-height:1.55;
    background:var(--surface-2); border:1px solid var(--line); border-radius:10px;
    padding:15px 17px; margin:0; overflow-x:auto; color:var(--ink);
    white-space:pre; tab-size:2;
  }
  pre.code code{ font-size:inherit; background:none; border:0; padding:0;
                 white-space:pre; color:inherit }
  .codewrap .lang{
    position:absolute; top:0; right:12px; transform:translateY(-50%);
    font-family:var(--font-mono); font-size:10px; letter-spacing:.1em;
    text-transform:uppercase; font-weight:600; color:var(--ink-3);
    background:var(--bg); border:1px solid var(--line); border-radius:20px;
    padding:2px 9px;
  }

  /* Pygments tokens, mapped onto the house palette so both themes work */
  pre.code .c,  pre.code .c1, pre.code .cm, pre.code .cs,
  pre.code .ch, pre.code .cpf{ color:var(--ink-3); font-style:italic }
  pre.code .sd{ color:var(--ink-3) }
  pre.code .k,  pre.code .kn, pre.code .kc, pre.code .kd,
  pre.code .kp, pre.code .kr, pre.code .kt, pre.code .ow{ color:var(--accent) }
  :root[data-theme="dark"] pre.code .k,
  :root[data-theme="dark"] pre.code .kn,
  :root[data-theme="dark"] pre.code .kc,
  :root[data-theme="dark"] pre.code .kd,
  :root[data-theme="dark"] pre.code .ow{ color:var(--accent-2) }
  @media(prefers-color-scheme:dark){
    pre.code .k, pre.code .kn, pre.code .kc,
    pre.code .kd, pre.code .ow{ color:var(--accent-2) }
  }
  pre.code .s,  pre.code .s1, pre.code .s2, pre.code .sa,
  pre.code .sb, pre.code .se, pre.code .si, pre.code .sr,
  pre.code .ss, pre.code .sx{ color:var(--ok) }
  pre.code .m,  pre.code .mi, pre.code .mf, pre.code .mh,
  pre.code .mo, pre.code .il{ color:var(--ok) }
  pre.code .nf, pre.code .fm{ color:var(--ink); font-weight:600 }
  pre.code .nc, pre.code .ne{ color:var(--warn); font-weight:600 }
  pre.code .nd{ color:var(--warn) }
  pre.code .nb, pre.code .bp{ color:var(--ink-2) }
  pre.code .nn, pre.code .nv, pre.code .no, pre.code .na{ color:var(--ink) }
  pre.code .o,  pre.code .p{ color:var(--ink-2) }
  pre.code .gd{ color:var(--gap) }
  pre.code .gi{ color:var(--ok) }
  pre.code .gh, pre.code .gu{ color:var(--ink-3); font-weight:600 }
  pre.code .err{ color:inherit; border:0 }

  /* blockquote -> callout */
  blockquote{
    margin:1.3em 0; padding:15px 18px; max-width:var(--prose);
    border:1px solid var(--line); border-left:3px solid var(--ink-3);
    border-radius:10px; background:var(--surface); color:var(--ink-2);
    font-size:15px;
  }
  blockquote p{ margin:.25em 0 }
  blockquote.warn{ border-left-color:var(--warn); background:var(--warn-bg) }

  /* figures / diagrams */
  figure.dgfig{ margin:1.6em 0 1.9em; max-width:none }
  .dgscroll{ overflow-x:auto; border:1px solid var(--line); border-radius:12px;
             background:var(--surface); padding:18px 16px }
  svg.dg{ max-width:100%; height:auto; display:block; margin:0 auto }
  svg.dg text{ font-family:var(--font-mono); fill:var(--ink); font-size:12px }
  svg.dg .t-sm{ font-size:11px; fill:var(--ink-2) }
  svg.dg .t-lbl{ font-size:10.5px; fill:var(--ink-3) }
  svg.dg .t-bad{ font-size:10.5px; fill:var(--gap); font-weight:600 }
  svg.dg .t-hd{ font-size:10.5px; font-weight:700; fill:var(--accent);
                letter-spacing:.09em }
  svg.dg .t-x{ font-size:15px; fill:var(--gap); font-weight:700 }
  svg.dg .bx{ fill:var(--surface); stroke:var(--line-2); stroke-width:1.2 }
  svg.dg .badbx{ fill:var(--gap-bg); stroke:var(--gap-line); stroke-width:1.3 }
  svg.dg .warnbx{ fill:var(--warn-bg); stroke:var(--warn-line); stroke-width:1.3 }
  svg.dg .hl{ fill:var(--ok-bg); stroke:var(--ok); stroke-width:1.6 }
  svg.dg .dead{ fill:var(--surface-2); stroke:var(--ink-3); stroke-width:1.2;
                stroke-dasharray:5 4; opacity:.8 }
  svg.dg .band{ fill:var(--surface-2); stroke:var(--line); stroke-width:1 }
  svg.dg .ok-band{ fill:var(--ok-bg); stroke:var(--ok-line) }
  svg.dg .bad-band{ fill:var(--gap-bg); stroke:var(--gap-line) }
  svg.dg .loopbx{ fill:none; stroke:var(--line-2); stroke-width:1;
                  stroke-dasharray:4 3 }
  svg.dg .life{ stroke:var(--line-2); stroke-width:1; stroke-dasharray:4 4 }
  svg.dg .ln{ stroke:var(--ink-2); stroke-width:1.4; fill:none }
  svg.dg .ln-bad{ stroke:var(--gap); stroke-width:1.6; stroke-dasharray:6 4; fill:none }
  svg.dg .ln-dead{ stroke:var(--ink-3); stroke-width:1.4; stroke-dasharray:5 4; fill:none }
  svg.dg .ln-soft{ stroke:var(--ink-3); stroke-width:1.2; stroke-dasharray:3 3; fill:none }
  svg.dg .mk{ fill:var(--ink-2); stroke:none }
  svg.dg .mk-bad{ fill:var(--gap); stroke:none }
  svg.dg .mk-dead{ fill:var(--ink-3); stroke:none }
  svg.dg .mk-soft{ fill:var(--ink-3); stroke:none }
  figure.dgfig figcaption{
    font-size:13.5px; color:var(--ink-2); line-height:1.5; margin-top:11px;
    max-width:var(--prose); padding-left:2px;
  }
  figure.dgfig figcaption b{ color:var(--ink) }

  /* severity pills in the findings tables */
  .g{ font-family:var(--font-mono); font-size:10.5px; font-weight:700;
      padding:1px 7px; border-radius:20px; border:1px solid; white-space:nowrap }
  .g.p0{ color:var(--gap); background:var(--gap-bg); border-color:var(--gap-line) }
  .g.p1{ color:var(--warn); background:var(--warn-bg); border-color:var(--warn-line) }
  .g.p2{ color:var(--ink-3); background:var(--surface-2); border-color:var(--line-2) }

  /* nested TOC */
  nav.toc ol ol{ list-style:none; counter-reset:none; margin:2px 0 6px;
                 padding-left:14px }
  nav.toc ol ol li{ counter-increment:none }
  nav.toc ol ol a{ padding:3px 8px 3px 10px; font-size:11.5px;
                   color:var(--ink-3); border-left:1px solid var(--line) }
  nav.toc ol ol a::before{ content:none }
  nav.toc ol ol a:hover{ color:var(--ink-2) }

  @media print{
    nav.toc{ display:none }
    section{ break-inside:avoid-page }
    .dgscroll{ overflow:visible }
    pre.code{ white-space:pre-wrap; word-break:break-word }
    body{ background:#fff; font-size:11pt }
  }
"""

TOC_SCRIPT = """
<script>
  // TOC active-state only. No animation; reduced-motion respected via CSS.
  // getElementById, not querySelector: several ids begin with a digit, which
  // is legal HTML but an invalid CSS selector.
  (function(){
    var links = Array.prototype.slice.call(document.querySelectorAll('nav.toc a'));
    var secs  = links.map(function(a){
      return document.getElementById(a.getAttribute('href').slice(1));
    });
    if(!('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(e){
        if(e.isIntersecting){
          var i = secs.indexOf(e.target);
          links.forEach(function(l){ l.style.background=''; l.style.color=''; });
          if(i>-1){ var l=links[i]; l.style.background='var(--surface-2)'; l.style.color='var(--ink)'; }
        }
      });
    },{rootMargin:'-10% 0px -80% 0px'});
    secs.forEach(function(s){ if(s) io.observe(s); });
  })();
</script>
"""


# --------------------------------------------------------------------------- #
# slugs
# --------------------------------------------------------------------------- #

def gh_slug(value: str, separator: str = "-") -> str:
    """GitHub-compatible heading slug.

    Matches the anchors the markdown sources already link to
    (e.g. '#5-approval-checklist', '#41-hermes-mission-flow-the-real-path'),
    so cross-document links keep working in the HTML build.
    """
    value = re.sub(r"<[^>]+>", "", value)
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.U)
    # Underscores survive — GitHub keeps them, so '5. `hermes_adapters/`'
    # slugs to '5-hermes_adapters'.
    #
    # Each whitespace character maps to one separator; runs are NOT collapsed.
    # github-slugger strips punctuation without closing the gap, so a removed
    # em dash or arrow leaves two spaces and therefore a double hyphen —
    # '4. Problem → solution traceability' is '4-problem--solution-traceability'.
    # Collapsing here would silently break every hand-written cross-link.
    return re.sub(r"\s", separator, value).strip(separator)


def split_heading(heading: str) -> Tuple[str, str, Optional[str]]:
    """'3. Layer model {#x}' -> ('03', 'Layer model', 'x')."""
    explicit = None
    m = re.search(r"\s*\{#([\w-]+)\}\s*$", heading)
    if m:
        explicit = m.group(1)
        heading = heading[: m.start()].strip()
    num = ""
    m = re.match(r"^(\d+)\.\s*(.*)$", heading)
    if m:
        num, heading = m.group(1).zfill(2), m.group(2)
    return num, heading, explicit


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def split_document(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Return (metadata lines, [(anchor, heading, body_markdown), ...])."""
    lines = text.splitlines()

    meta: List[str] = []
    i = 0
    while i < len(lines) and not lines[i].startswith("# "):
        i += 1
    i += 1
    while i < len(lines) and not lines[i].startswith("---"):
        if lines[i].strip():
            meta.append(lines[i].strip())
        i += 1

    sections: List[Tuple[str, str, str]] = []
    cur: Optional[str] = None
    buf: List[str] = []
    for line in lines[i:]:
        if line.startswith("## "):
            if cur is not None:
                sections.append((_sec_anchor(cur), cur, "\n".join(buf)))
            cur, buf = line[3:].strip(), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        sections.append((_sec_anchor(cur), cur, "\n".join(buf)))
    return meta, sections


def _sec_anchor(heading: str) -> str:
    _num, title, explicit = split_heading(heading)
    if explicit:
        return explicit
    # slug the heading as written (number included), matching GitHub
    return gh_slug(re.sub(r"\s*\{#[\w-]+\}\s*$", "", heading))


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

MD = markdown.Markdown(
    extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
    extension_configs={"toc": {"slugify": gh_slug}},
)

_MERMAID_TOKEN = "@@DIAGRAM-%d@@"
_CODE_TOKEN = "@@CODE-%d@@"

LEXER_ALIAS = {"jsonc": "json", "gitignore": "text", "": "text"}


def extract_fences(body: str) -> Tuple[str, List[Tuple[str, str]]]:
    blocks: List[Tuple[str, str]] = []

    def repl(m: re.Match) -> str:
        lang, code = (m.group(1) or "").strip(), m.group(2)
        blocks.append((lang, code))
        token = (_MERMAID_TOKEN if lang == "mermaid" else _CODE_TOKEN) % (len(blocks) - 1)
        return f"\n\n{token}\n\n"

    body = re.sub(r"```([a-zA-Z0-9+_-]*)\n(.*?)```", repl, body, flags=re.S)
    return body, blocks


def render_code(lang: str, code: str) -> str:
    """Pygments-tokenised block with a language chip.

    ``nowrap`` keeps Pygments' own <div>/<pre> out of the way — the house
    stylesheet owns the container, and the token colours are remapped onto
    theme variables so light and dark both read correctly.
    """
    code = code.rstrip("\n")
    name = LEXER_ALIAS.get(lang, lang)
    try:
        lexer = get_lexer_by_name(name)
    except ClassNotFound:
        lexer = TextLexer()
    if isinstance(lexer, TextLexer):
        inner = html.escape(code)
    else:
        inner = highlight(code, lexer, HtmlFormatter(nowrap=True)).rstrip("\n")
    wide = max((len(l) for l in code.splitlines()), default=0) > 78
    chip = f'<span class="lang">{html.escape(lang or "text")}</span>'
    cls = "codewrap wide" if wide else "codewrap"
    return f'<div class="{cls}">{chip}<pre class="code"><code>{inner}</code></pre></div>'


GRADE_CLASS = {"P0": "gap", "P1": "warn", "P2": "plain"}


def post_process(chunk: str) -> str:
    """Tighten the raw markdown output into the house idiom."""
    chunk = re.sub(r"<table>", '<div class="tablewrap"><table class="wide">', chunk)
    chunk = re.sub(r"</table>", "</table></div>", chunk)

    # grade cells -> pills
    for grade in ("P0", "P1", "P2"):
        chunk = chunk.replace(
            f"<td><strong>{grade}</strong></td>",
            f'<td><span class="g {grade.lower()}">{grade}</span></td>')

    # "**Category:** CORR · **Grade:** P0" -> a pill row
    def _pillrow(m: re.Match) -> str:
        cat, grade = m.group(1).strip(), m.group(2)
        pc = GRADE_CLASS.get(grade, "plain")
        return (f'<p class="pillrow">'
                f'<span class="pill plain"><span class="dot"></span>{cat}</span>'
                f'<span class="pill {pc}"><span class="dot"></span>{grade}</span>'
                f'</p>')

    chunk = re.sub(
        r"<p><strong>Category:</strong>\s*(.*?)\s*·\s*"
        r"<strong>Grade:</strong>\s*(P\d)\s*</p>", _pillrow, chunk)

    # leading finding id in an h3 -> mono badge; inline (P0) -> pill.
    # Docs that write '### F-01 — …' without an explicit {#f-01} anchor still
    # get a '#f-01' target, so the short cross-references used throughout the
    # register resolve regardless of which convention a document follows.
    def _h3(m: re.Match) -> str:
        attrs, text = m.group(1), m.group(2)
        alias = ""
        code = re.match(r"^([A-Z]{1,3}-\d{2,3})\b", text)
        if code:
            want = code.group(1).lower()
            have = re.search(r'id="([^"]+)"', attrs)
            if not have or have.group(1) != want:
                alias = f'<span class="anchor-alias" id="{want}"></span>'
        text = re.sub(r"^([A-Z]{1,3}-\d{2,3})\s*", r'<span class="fid">\1</span> ', text)
        text = re.sub(
            r"^(<span class=\"fid\">[^<]+</span>\s*)\((P\d)\)\s*",
            lambda g: g.group(1) + f'<span class="g {g.group(2).lower()}">'
                                   f'{g.group(2)}</span> ', text)
        text = re.sub(r"^\s*—\s*", "", text)
        text = re.sub(r"(</span>)\s*—\s*", r"\1 ", text)
        return f"{alias}<h3{attrs}>{text}</h3>"

    chunk = re.sub(r"<h3([^>]*)>(.*?)</h3>", _h3, chunk, flags=re.S)

    chunk = chunk.replace("<blockquote>\n<p>⚠", '<blockquote class="warn">\n<p>⚠')
    return chunk


def render_body(body: str, diagram_fns: Sequence, cursor: List[int]) -> str:
    body, blocks = extract_fences(body)
    chunk = MD.reset().convert(body)
    chunk = post_process(chunk)

    for idx, (lang, code) in enumerate(blocks):
        if lang == "mermaid":
            token = _MERMAID_TOKEN % idx
            replacement = diagram_fns[cursor[0]]()
            cursor[0] += 1
        else:
            token = _CODE_TOKEN % idx
            replacement = render_code(lang, code)
        chunk = chunk.replace(f"<p>{token}</p>", replacement).replace(token, replacement)
    return chunk


def rewrite_links(chunk: str) -> str:
    for md_name, html_name in LINK_REWRITES.items():
        chunk = chunk.replace(md_name, html_name)
    return chunk


def sub_toc(chunk: str) -> List[Tuple[str, str]]:
    """Collect (id, text) for every h3 in a rendered section."""
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r'<h3 id="([^"]+)"[^>]*>(.*?)</h3>', chunk, flags=re.S):
        text = re.sub(r"<[^>]+>", "", m.group(2))
        text = html.unescape(text).strip()
        out.append((m.group(1), text))
    return out


def render_meta(meta: Sequence[str]) -> str:
    out: List[str] = []
    for line in meta:
        rendered = rewrite_links(MD.reset().convert(line))
        rendered = re.sub(r"</?p>", "", rendered).strip()
        for part in rendered.split(" · "):
            if part.strip():
                out.append(f"<span>{part.strip()}</span>")
    return "\n      ".join(out)


def build(src: Path, out_dir: Path) -> Path:
    key = src.name
    front = FRONT[key]
    meta, sections = split_document(src.read_text(encoding="utf-8"))

    diagram_fns = DIAGRAMS.get(key, [])
    cursor = [0]

    toc: List[str] = []
    body: List[str] = []
    for anchor, heading, raw in sections:
        num, title, _ = split_heading(heading)
        rendered = rewrite_links(render_body(raw, diagram_fns, cursor))

        subs = sub_toc(rendered)
        sub_html = ""
        if len(subs) >= 2:
            items = "".join(
                f'<li><a href="#{sid}">{html.escape(stext)}</a></li>'
                for sid, stext in subs)
            sub_html = f"<ol>{items}</ol>"
        toc.append(f'<li><a href="#{anchor}">{html.escape(title)}</a>{sub_html}</li>')

        eyebrow = f"§&nbsp;{num}" if num else "&nbsp;"
        body.append(
            f'      <section id="{anchor}">\n'
            f'        <div class="sec-eyebrow">{eyebrow} · {html.escape(title)}</div>\n'
            f'        <h2>{html.escape(title)}</h2>\n'
            f'{rendered}\n'
            f'      </section>'
        )

    assert cursor[0] == len(diagram_fns), (
        f"{key}: consumed {cursor[0]} of {len(diagram_fns)} diagrams")

    css = HOUSE_CSS.read_text(encoding="utf-8").replace("</style>", EXTRA_CSS + "\n</style>")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{re.sub('&nbsp;', ' ', front['title'])}</title>
<meta name="description" content="{html.escape(re.sub('<[^>]+>', '', front['deck']))}">
{css}
</head>
<body>
<div class="shell">
  <header class="mast">
    <div class="eyebrow">
      <span>{front['eyebrow']}</span><span class="sep">/</span>
      <span>{front['eyebrow2']}</span><span class="sep">/</span>
      <span>{front['eyebrow3']}</span>
    </div>
    <h1 class="title">{front['title']}</h1>
    <p class="deck">{front['deck']}</p>
    <div class="metabar">
      {render_meta(meta)}
    </div>
  </header>

  <div class="body-grid">
    <nav class="toc" aria-label="Contents">
      <p class="toc-h">Contents</p>
      <ol>
        {chr(10).join('        ' + t for t in toc).strip()}
      </ol>
    </nav>

    <main>
{chr(10).join(body)}

      <div class="foot">
        <p class="mono">Generated from {html.escape(key)}. File:line references are to
        the repository as read on 2026-08-10 (main @ 35e2eeb); verify against current
        source before acting. This document is a review artefact — no project code was
        modified to produce it.</p>
      </div>
    </main>
  </div>
</div>
{TOC_SCRIPT}
</body>
</html>
"""
    dest = out_dir / front["out"]
    dest.write_text(doc, encoding="utf-8")
    return dest


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for src, out_dir in SOURCES:
        dest = build(src, out_dir)
        print(f"  {dest.relative_to(REPO)}  ({dest.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
