# Architecture review — tooling

Scripts that produced the August 2026 architecture review and that rebuild its HTML
deliverables. Everything here is **read-only with respect to the codebase under review**:
nothing in this folder modifies `hermes/`, `Config/`, `App/`, or `experiments/`.

---

## One command

```bash
python "DeveloperDocs/architecture review/build.py"
```

Renders all ten documents, checks every diagram's layout, verifies every HTML file, and
exits non-zero if anything fails. Runnable from any working directory — paths are derived
from `__file__`, not from the CWD.

---

## What gets built

| Source (Markdown) | Output (HTML) |
|---|---|
| `architecture documents/System_Architecture_Overview.md` | `…/System_Architecture_Overview.html` |
| `architecture documents/Hermes/HERMES_Architecture.md` | `…/HERMES_Architecture.html` |
| `architecture documents/HIFins/HIFINS_Architecture.md` | `…/HIFINS_Architecture.html` |
| `Codebase Review/00_Critical_Problem_Areas.md` | `…/00_Critical_Problem_Areas.html` |
| `Codebase Review/01_Refactoring_Strategy.md` | `…/01_Refactoring_Strategy.html` |
| `Codebase Review/Hermes/HERMES_Findings_and_Refactoring.md` | `…/HERMES_Findings_and_Refactoring.html` |
| `Codebase Review/Hermes/HERMES_Production_Code.md` | `…/HERMES_Production_Code.html` |
| `Codebase Review/HIFINS/HIFINS_Findings_and_Refactoring.md` | `…/HIFINS_Findings_and_Refactoring.html` |
| `Codebase Review/HIFINS/HIFINS_Production_Code.md` | `…/HIFINS_Production_Code.html` |
| `Comparative Analysis/CEDA_vs_HERMES.md` | `…/CEDA_vs_HERMES.html` |

**Markdown is the source of truth.** Edit the `.md`, re-run `build.py`, commit both.
Never hand-edit the generated `.html` — the next build overwrites it.

Output is fully self-contained: embedded CSS, inline SVG, no network requests, no runtime
dependencies. The files open from `file://`, survive being emailed, and print sensibly.

---

## Files

### Rendering

| File | Role |
|---|---|
| `build.py` | Entry point — render, then both checks |
| `render_docs.py` | Markdown → HTML. Owns the per-document front matter (`FRONT`), the source list (`SOURCES`), section/anchor slugging, code highlighting, and the TOC |
| `diagrams.py` | Hand-authored inline SVG, D1–D6, plus the drawing primitives (`box`, `band`, `arrow`, `label`, `figure`) |
| `diagrams_cmp.py` | Inline SVG D7–D8 for the comparative analysis |
| `house.css.html` | The house stylesheet, lifted from `HERMES_Experiment4_Integrated_Design_and_Plan.html` so every generated document matches the existing DeveloperDocs look |

### Verification

| File | Role |
|---|---|
| `check_svg.py` | Analytic layout check on every registered diagram — clipping past the `viewBox`, text-vs-text collisions, text straddling a box border. Computes bounding boxes from monospace advance widths, so it needs no browser |
| `verify_html.py` | Per-file: tag balance, conversion leftovers, required `<head>` structure, self-containment, `role="img"` + `aria-label` on every SVG, unique ids, no dangling marker refs. Across files: every same-page and cross-document `#anchor` resolves |

Both are derived from the renderer's own registries, so a new document or diagram is
covered automatically once it is registered — there is no second list to keep in sync.

### Analysis (produced the review's findings)

| File | Role |
|---|---|
| `audit.py` | Repo-wide AST audit: function length, cyclomatic complexity, parameter counts, exception shape, thread/sleep sites, cross-object private access, TODO markers. Source of the statistics in `00_Critical_Problem_Areas.md` |
| `undef.py` | Names referenced but never bound in a module. Found the missing `Sequence` import in `host_mission.py` (finding Q-01) |
| `read_pptx.py` | Slide text, tables, charts, notes, and picture inventory from a `.pptx`. Used for the CEDA comparison |

```bash
python "DeveloperDocs/architecture review/audit.py"
python "DeveloperDocs/architecture review/undef.py"
python "DeveloperDocs/architecture review/read_pptx.py" deck.pptx > dump.txt
```

Re-run `audit.py` after a refactoring phase lands to confirm the counts moved the way the
plan said they would.

---

## Adding a document

1. Write the Markdown anywhere under `DeveloperDocs/`.
2. Add a `FRONT` entry in `render_docs.py` — `out`, three `eyebrow` strings, `title`, `deck`.
3. Add an entry to `SOURCES` — `(source_path, output_dir)`.
4. Run `build.py`.

Verification picks it up automatically.

## Adding a diagram

1. Write a function in `diagrams.py` (or `diagrams_cmp.py`) returning `figure(...)`.
2. Register it in the `DIAGRAMS` dict against the Markdown filename, in the order the
   ` ```mermaid ` fences appear in that file.
3. Run `build.py` — `check_svg.py` will report any clipping or collision before you
   ever open the page.

Diagram conventions: size by `viewBox`, colour only through the CSS classes defined in
`render_docs.EXTRA_CSS` (never literal hex, so light and dark both work), arrowheads via
`<marker>` with a per-diagram id prefix, and every figure carries `role="img"`, an
`aria-label` stating the mechanism, and a `<figcaption>` stating the one claim it makes.

---

## Dependencies

`markdown`, `pygments` for rendering; `python-pptx` for `read_pptx.py` only. `audit.py`,
`undef.py`, `check_svg.py` and `verify_html.py` are pure standard library.

```bash
pip install markdown pygments python-pptx
```

---

## Known environment note

`markitdown` is unusable in this environment — the installed pandas is built against
NumPy 1.x while NumPy 2.4 is present, so importing it raises a binary-incompatibility
`ValueError`. `read_pptx.py` exists because of that and reads the deck through
`python-pptx` instead. The same ABI mismatch is why four `tests/unit` modules fail
locally; see finding T-01.
