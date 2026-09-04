#!/usr/bin/env python3
# Script:      render_deck_pdf.py
# Class:       B - Deterministic build step producing a submission artefact
# Version:     2026.1.0.0
# Author:      Hadrian Hu
# Date:        2026-09-03
# Description: Renders the X Quant X HTML presentation deck to a paginated,
#              print-background PDF at exactly 1920x1080 CSS pixels per page,
#              and optionally exports every slide as a full-HD PNG.
# Why:         The LabLab submission requires a PDF deck. Producing it from the
#              same HTML that drives the recorded video guarantees the PDF and
#              the video can never drift apart, which a hand-built PDF cannot.
# How:         Launches headless Chromium via Playwright, navigates to the local
#              deck, waits for fonts to settle, then calls page.pdf() with print
#              media emulation so theme.css's @media print rules paginate one
#              slide per page.
# Usage:       python scripts/presentation/render_deck_pdf.py [--output-file PATH]
#                  [--deck PATH] [--png-dir PATH] [--dry-run] [--verbose] [--version]
# Depends:     playwright>=1.55 with the chromium browser installed,
#              presentation/deck/index.html, presentation/deck/theme.css
# Output:      presentation/output/xquantx_deck.pdf (default); optional PNG set.
#              Exit codes: 0 success, 1 failure, 2 bad args, 3 missing dependency,
#              4 dry-run complete.
# Ref:         coding_stds/scripting/python_scripting_standards.txt SS2, SS3, SS4, SS7

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _deck_common import (  # noqa: E402  (requires the sys.path shim above)
    DECK_INDEX,
    EXIT_BAD_ARGS,
    EXIT_DRY_RUN,
    EXIT_FAILURE,
    EXIT_MISSING_DEPENDENCY,
    EXIT_OK,
    OUTPUT_DIR,
    SLIDE_HEIGHT_PX,
    SLIDE_WIDTH_PX,
    configure_logging,
    deck_url,
    iter_slides,
    require_playwright,
)

__version__ = "2026.1.0.0"

DEFAULT_PDF_NAME = "xquantx_deck.pdf"
FONT_SETTLE_MS = 1500


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # WHAT: Parses and validates CLI arguments for the PDF render.
    # WHY:  SS4.1 forbids manual sys.argv parsing; SS4.6 mandates the standard
    #       --dry-run / --verbose / --version trio on Class B scripts.
    # HOW:  argparse with long-form flags and a help string on every argument.
    parser = argparse.ArgumentParser(
        description="Render the X Quant X HTML deck to a full-HD paginated PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--deck",
        type=Path,
        default=DECK_INDEX,
        help=f"Path to the deck entry HTML file. Default: {DECK_INDEX}",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_DIR / DEFAULT_PDF_NAME,
        help=f"Destination PDF path. Default: presentation/output/{DEFAULT_PDF_NAME}",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="If set, also write one full-HD PNG per slide into this directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report the intended actions without rendering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level log output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"render_deck_pdf.py {__version__}",
    )
    return parser.parse_args(argv)


def validate_inputs(deck: Path, output_file: Path) -> None:
    # WHAT: Confirms the deck exists and the output directory is creatable.
    # WHY:  A missing deck must fail with exit code 2, not a Playwright timeout
    #       several seconds later with an opaque navigation error.
    # HOW:  Explicit existence check plus mkdir(parents=True, exist_ok=True).
    if not deck.is_file():
        raise FileNotFoundError(f"Deck entry file not found: {deck}")
    output_file.parent.mkdir(parents=True, exist_ok=True)


def render(deck: Path, output_file: Path, png_dir: Optional[Path]) -> int:
    # WHAT: Drives headless Chromium to produce the PDF and optional PNG set.
    # WHY:  Isolating the browser session from argument handling keeps main()
    #       to a call sequence, per SS3.4.
    # HOW:  page.pdf() with print media emulation honours theme.css @media print,
    #       which forces every .slide visible and paginated one per page.
    from playwright.sync_api import sync_playwright

    logger = configure_logging(verbose=False)
    written = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--force-color-profile=srgb"])
        try:
            page = browser.new_page(
                viewport={"width": SLIDE_WIDTH_PX, "height": SLIDE_HEIGHT_PX},
                device_scale_factor=2,
            )
            page.goto(deck_url(deck), wait_until="networkidle")
            page.wait_for_timeout(FONT_SETTLE_MS)

            if png_dir is not None:
                png_dir.mkdir(parents=True, exist_ok=True)
                for index in iter_slides(page):
                    target = png_dir / f"slide_{index + 1:02d}.png"
                    page.screenshot(path=str(target))
                    logger.debug("Wrote slide image: %s", target)
                    written += 1
                page.evaluate("() => window.XQX.goto(0)")

            page.pdf(
                path=str(output_file),
                width=f"{SLIDE_WIDTH_PX}px",
                height=f"{SLIDE_HEIGHT_PX}px",
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                prefer_css_page_size=True,
            )
        finally:
            browser.close()

    logger.info("PDF written: %s (%.1f KB)", output_file, output_file.stat().st_size / 1024)
    if png_dir is not None:
        logger.info("Slide images written: %d into %s", written, png_dir)
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(args.verbose)
    try:
        require_playwright()
        validate_inputs(args.deck, args.output_file)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return EXIT_MISSING_DEPENDENCY
    except (FileNotFoundError, OSError) as exc:
        logger.error("%s", exc)
        return EXIT_BAD_ARGS
    if args.dry_run:
        logger.info("Dry run: would render %s -> %s", args.deck, args.output_file)
        return EXIT_DRY_RUN
    try:
        return render(args.deck, args.output_file, args.png_dir)
    except Exception as exc:  # noqa: BLE001 - top-level boundary, logged then coded
        logger.error("Render failed: %s", exc)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
