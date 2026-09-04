#!/usr/bin/env python3
# Script:      _deck_common.py
# Class:       A - Shared helpers for the presentation media scripts
# Version:     2026.1.0.0
# Author:      Hadrian Hu
# Date:        2026-09-03
# Description: Provides the repository path constants, logging configuration,
#              exit-code constants and Playwright deck-driving helpers shared by
#              render_deck_pdf.py, record_deck_video.py and
#              capture_dashboard_media.py.
# Why:         Without a shared module each presentation script would duplicate
#              its own path resolution, logging setup and slide-advance loop,
#              which is exactly the drift that python_scripting_standards.txt
#              SS9 and the DRY principle exist to prevent.
# How:         Exposes pure, side-effect-free constants and small typed helper
#              functions; importing this module performs no I/O beyond resolving
#              paths, per aesthetic_standards.txt P.5.2.
# Usage:       Imported by the sibling scripts. Not directly executable.
# Depends:     playwright>=1.55 (only for the type-hinted helpers), Python 3.11+
# Output:      None. Pure helper module.
# Ref:         coding_stds/scripting/python_scripting_standards.txt

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover - import used for annotations only
    from playwright.sync_api import Page

__version__ = "2026.1.0.0"

# --- Repository geometry ---------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PRESENTATION_DIR: Path = REPO_ROOT / "presentation"
DECK_DIR: Path = PRESENTATION_DIR / "deck"
DECK_INDEX: Path = DECK_DIR / "index.html"
OUTPUT_DIR: Path = PRESENTATION_DIR / "output"

# --- Deck geometry (16:9 at full HD, matching theme.css --slide-w/h) -------

SLIDE_WIDTH_PX: int = 1920
SLIDE_HEIGHT_PX: int = 1080

# --- Exit codes (python_scripting_standards.txt SS7.2) ---------------------

EXIT_OK: int = 0
EXIT_FAILURE: int = 1
EXIT_BAD_ARGS: int = 2
EXIT_MISSING_DEPENDENCY: int = 3
EXIT_DRY_RUN: int = 4


def configure_logging(verbose: bool) -> logging.Logger:
    # WHAT: Installs the project's standard log format at INFO or DEBUG.
    # WHY:  SS6.1 requires logging to be configured before any logic executes.
    # HOW:  logging.basicConfig with force=True so repeated calls in tests are
    #       idempotent rather than silently ignored.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        force=True,
    )
    return logging.getLogger("xquantx.presentation")


def deck_url(index_path: Path) -> str:
    # WHAT: Converts a deck path into a file:// URL Playwright can navigate to.
    # WHY:  Windows drive-letter paths are not valid URLs; as_uri() handles the
    #       percent-encoding and slash direction correctly.
    # HOW:  Path.resolve().as_uri().
    return index_path.resolve().as_uri()


def require_playwright() -> None:
    # WHAT: Fails fast with an actionable message if Playwright is unavailable.
    # WHY:  SS7.2 reserves exit code 3 for a missing dependency; an ImportError
    #       traceback is not an actionable message for a CI operator.
    # HOW:  Import inside the function so module import stays side-effect free.
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "playwright is not installed in the active interpreter. "
            "Run: python -m pip install playwright && python -m playwright install chromium"
        ) from exc


def slide_count(page: "Page") -> int:
    # WHAT: Reads the deck's own slide count through its public JS API.
    # WHY:  Reading window.XQX rather than counting DOM nodes keeps the scripts
    #       coupled to the deck's declared interface, not its markup.
    # HOW:  page.evaluate against the API installed by deck.js.
    return int(page.evaluate("() => window.XQX.count()"))


def iter_slides(page: "Page") -> Iterator[int]:
    # WHAT: Yields each zero-based slide index after activating that slide.
    # WHY:  Both the video recorder and any future frame exporter need the same
    #       deterministic traversal; duplicating it invites divergence.
    # HOW:  Drives window.XQX.goto, which is the same path a keyboard user takes.
    total = slide_count(page)
    for index in range(total):
        page.evaluate("(i) => window.XQX.goto(i)", index)
        yield index
