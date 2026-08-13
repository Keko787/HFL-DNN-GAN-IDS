"""Build and verify every review document in one command.

    python "DeveloperDocs/architecture review/build.py"

Runs, in order:

  1. render_docs  — markdown -> self-contained HTML (10 documents)
  2. check_svg    — analytic layout check on every hand-authored diagram
  3. verify_html  — structure, self-containment, and anchor integrity

Exits non-zero if either check reports a problem, so this is safe to wire
into CI or a pre-commit hook.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import check_svg  # noqa: E402
import render_docs  # noqa: E402
import verify_html  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rule("1/3  RENDER")
    render_docs.main()

    rule("2/3  DIAGRAM LAYOUT")
    svg_rc = check_svg.main()

    rule("3/3  HTML STRUCTURE")
    html_rc = verify_html.main()

    rule("RESULT")
    if svg_rc or html_rc:
        print("FAILED — see the reports above.")
        return 1
    print(f"OK — {len(verify_html.FILES)} documents rendered and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
