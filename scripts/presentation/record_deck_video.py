#!/usr/bin/env python3
# Script:      record_deck_video.py
# Class:       B - Deterministic build step producing a submission artefact
# Version:     2026.1.0.0
# Author:      Hadrian Hu
# Date:        2026-09-03
# Description: Records a long-form, full-HD (1920x1080) walkthrough video of the
#              X Quant X presentation deck by driving it slide by slide in a
#              Playwright-instrumented Chromium session with video capture on,
#              then optionally transcodes the WebM capture to H.264 MP4.
# Why:         The LabLab submission requires a high-definition video of the same
#              material as the PDF. Recording the live HTML deck rather than
#              stitching exported images keeps the video and the PDF provably in
#              sync and preserves crisp vector text at full resolution.
# How:         Opens a browser context with record_video_dir and a 1920x1080
#              record_video_size, walks window.XQX.goto over every slide with a
#              configurable dwell time, closes the context so Playwright flushes
#              the video file, renames it to the requested output path, and, when
#              --mp4 is passed, transcodes it with the bundled imageio-ffmpeg
#              binary.
# Usage:       python scripts/presentation/record_deck_video.py
#                  [--output-file PATH] [--seconds-per-slide FLOAT]
#                  [--intro-seconds FLOAT] [--outro-seconds FLOAT] [--mp4]
#                  [--deck PATH] [--dry-run] [--verbose] [--version]
# Depends:     playwright>=1.55 with chromium installed; imageio-ffmpeg only when
#              --mp4 is requested; presentation/deck/index.html
# Output:      presentation/output/xquantx_deck.webm (default), plus
#              xquantx_deck.mp4 when --mp4 is given.
#              Exit codes: 0 success, 1 failure, 2 bad args, 3 missing dependency,
#              4 dry-run complete.
# Ref:         coding_stds/scripting/python_scripting_standards.txt SS2, SS3, SS4, SS7

from __future__ import annotations

import argparse
import shutil
import subprocess
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
    slide_count,
)

__version__ = "2026.1.0.0"

DEFAULT_VIDEO_NAME = "xquantx_deck.webm"
DEFAULT_SECONDS_PER_SLIDE = 14.0
DEFAULT_INTRO_SECONDS = 4.0
DEFAULT_OUTRO_SECONDS = 6.0
FONT_SETTLE_MS = 1500


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # WHAT: Parses and validates CLI arguments for the video recording.
    # WHY:  Dwell timing is the one knob a presenter genuinely needs to tune, so
    #       it must be a flag rather than an edit to the script body.
    # HOW:  argparse with long-form flags and the standard --dry-run/--verbose trio.
    parser = argparse.ArgumentParser(
        description="Record a full-HD walkthrough video of the X Quant X deck.",
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
        default=OUTPUT_DIR / DEFAULT_VIDEO_NAME,
        help=f"Destination WebM path. Default: presentation/output/{DEFAULT_VIDEO_NAME}",
    )
    parser.add_argument(
        "--seconds-per-slide",
        type=float,
        default=DEFAULT_SECONDS_PER_SLIDE,
        help=f"Dwell time on each slide, in seconds. Default: {DEFAULT_SECONDS_PER_SLIDE}",
    )
    parser.add_argument(
        "--intro-seconds",
        type=float,
        default=DEFAULT_INTRO_SECONDS,
        help=f"Extra hold on the title slide. Default: {DEFAULT_INTRO_SECONDS}",
    )
    parser.add_argument(
        "--outro-seconds",
        type=float,
        default=DEFAULT_OUTRO_SECONDS,
        help=f"Extra hold on the closing slide. Default: {DEFAULT_OUTRO_SECONDS}",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Also transcode the capture to H.264 MP4 via the imageio-ffmpeg binary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the intended recording plan and duration without recording.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level log output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"record_deck_video.py {__version__}",
    )
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> None:
    # WHAT: Confirms the deck exists and all timing values are strictly positive.
    # WHY:  A zero or negative dwell silently produces an unusable one-frame
    #       video; failing at argument-validation time is far cheaper.
    # HOW:  Explicit checks raising ValueError/FileNotFoundError for exit code 2.
    if not args.deck.is_file():
        raise FileNotFoundError(f"Deck entry file not found: {args.deck}")
    for name in ("seconds_per_slide", "intro_seconds", "outro_seconds"):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0, got {value}")
    if float(args.seconds_per_slide) <= 0.0:
        raise ValueError("--seconds-per-slide must be greater than 0")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)


def transcode_to_mp4(source: Path) -> Path:
    # WHAT: Transcodes the WebM capture to a widely-playable H.264 MP4.
    # WHY:  WebM is not universally accepted by submission portals; MP4 is.
    # HOW:  Uses the ffmpeg binary bundled with imageio-ffmpeg so no separate
    #       system install is required, with CRF 18 to keep text edges sharp.
    from imageio_ffmpeg import get_ffmpeg_exe

    target = source.with_suffix(".mp4")
    command = [
        get_ffmpeg_exe(),
        "-y",
        "-i", str(source),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={SLIDE_WIDTH_PX}:{SLIDE_HEIGHT_PX}:flags=lanczos",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return target


def record(args: argparse.Namespace) -> int:
    # WHAT: Runs the instrumented browser session and produces the video file.
    # WHY:  Playwright only flushes a context video on context.close(), so the
    #       rename must happen after the context is closed - a real ordering
    #       constraint, not a stylistic one.
    # HOW:  record_video_dir into a staging folder, walk the slides, close, move.
    from playwright.sync_api import sync_playwright

    logger = configure_logging(args.verbose)
    staging = args.output_file.parent / "_capture"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--force-color-profile=srgb"])
        context = browser.new_context(
            viewport={"width": SLIDE_WIDTH_PX, "height": SLIDE_HEIGHT_PX},
            record_video_dir=str(staging),
            record_video_size={"width": SLIDE_WIDTH_PX, "height": SLIDE_HEIGHT_PX},
        )
        try:
            page = context.new_page()
            page.goto(deck_url(args.deck), wait_until="networkidle")
            page.wait_for_timeout(FONT_SETTLE_MS)
            total = slide_count(page)
            logger.info("Recording %d slides at %.1fs each.", total, args.seconds_per_slide)

            for index in range(total):
                page.evaluate("(i) => window.XQX.goto(i)", index)
                dwell = float(args.seconds_per_slide)
                if index == 0:
                    dwell += float(args.intro_seconds)
                if index == total - 1:
                    dwell += float(args.outro_seconds)
                logger.debug("Slide %d/%d holding %.1fs", index + 1, total, dwell)
                page.wait_for_timeout(int(dwell * 1000))
        finally:
            context.close()
            browser.close()

    captures = sorted(staging.glob("*.webm"))
    if not captures:
        raise RuntimeError(f"Playwright produced no video file in {staging}")
    args.output_file.unlink(missing_ok=True)
    captures[0].replace(args.output_file)
    shutil.rmtree(staging, ignore_errors=True)
    logger.info(
        "Video written: %s (%.1f MB)",
        args.output_file,
        args.output_file.stat().st_size / (1024 * 1024),
    )

    if args.mp4:
        mp4_path = transcode_to_mp4(args.output_file)
        logger.info(
            "MP4 written: %s (%.1f MB)", mp4_path, mp4_path.stat().st_size / (1024 * 1024)
        )
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(args.verbose)
    try:
        validate_inputs(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return EXIT_BAD_ARGS
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        logger.error(
            "playwright is not installed. Run: python -m pip install playwright "
            "&& python -m playwright install chromium"
        )
        return EXIT_MISSING_DEPENDENCY
    if args.dry_run:
        logger.info(
            "Dry run: would record %s -> %s at %.1fs per slide.",
            args.deck,
            args.output_file,
            args.seconds_per_slide,
        )
        return EXIT_DRY_RUN
    try:
        return record(args)
    except Exception as exc:  # noqa: BLE001 - top-level boundary, logged then coded
        logger.error("Recording failed: %s", exc)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
