#!/usr/bin/env python3
# Script:      capture_dashboard_media.py
# Class:       C - Multi-step orchestration with an external process dependency
# Version:     2026.1.0.0
# Author:      Hadrian Hu
# Date:        2026-09-03
# Description: Starts the X Quant X Dash control room in a child process (or
#              attaches to an already-running instance), then uses Playwright to
#              capture full-HD screenshots of it and, optionally, a recorded
#              walkthrough video for the submission media set.
# Why:         The LabLab submission requires evidence of a functional dashboard
#              showing P&L, decisions, margins and risk metrics. Capturing the
#              real running application is the only honest way to produce that
#              evidence; a mocked-up image would be a fabricated claim.
# How:         Launches scripts/run_dashboard.py with subprocess, polls the HTTP
#              endpoint until it answers, drives Playwright against it, then
#              terminates the child process it started. A dashboard the operator
#              started themselves (--attach) is never terminated by this script.
# Usage:       python scripts/presentation/capture_dashboard_media.py
#                  [--host HOST] [--port PORT] [--attach] [--record-video]
#                  [--video-seconds FLOAT] [--output-dir PATH]
#                  [--dry-run] [--verbose] [--version]
# Depends:     playwright>=1.55 with chromium installed; dash, plotly and the
#              investment_agent package importable; a free TCP port.
# Output:      presentation/output/dashboard/*.png and, with --record-video,
#              presentation/output/dashboard/xquantx_dashboard.webm.
#              Exit codes: 0 success, 1 failure, 2 bad args, 3 missing dependency,
#              4 dry-run complete, 20 the dashboard never became reachable.
# Ref:         coding_stds/scripting/python_scripting_standards.txt SS2, SS3, SS4, SS7

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _deck_common import (  # noqa: E402  (requires the sys.path shim above)
    EXIT_BAD_ARGS,
    EXIT_DRY_RUN,
    EXIT_FAILURE,
    EXIT_MISSING_DEPENDENCY,
    EXIT_OK,
    OUTPUT_DIR,
    REPO_ROOT,
    SLIDE_HEIGHT_PX,
    SLIDE_WIDTH_PX,
    configure_logging,
)

__version__ = "2026.1.0.0"

EXIT_SERVICE_UNREACHABLE = 20
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050
DEFAULT_VIDEO_SECONDS = 30.0
STARTUP_TIMEOUT_S = 60.0
STARTUP_POLL_S = 1.0
RENDER_SETTLE_MS = 4000


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # WHAT: Parses and validates CLI arguments for the dashboard capture.
    # WHY:  --attach exists so an operator can capture a session they have
    #       already configured, rather than forcing a cold start every time.
    # HOW:  argparse with long-form flags and the standard --dry-run/--verbose trio.
    parser = argparse.ArgumentParser(
        description="Capture screenshots and video of the running X Quant X dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Dashboard host. Default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Dashboard port. Default: {DEFAULT_PORT}")
    parser.add_argument(
        "--attach",
        action="store_true",
        help="Attach to an already-running dashboard instead of starting one.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Also record a walkthrough video of the dashboard.",
    )
    parser.add_argument(
        "--video-seconds",
        type=float,
        default=DEFAULT_VIDEO_SECONDS,
        help=f"Walkthrough duration in seconds. Default: {DEFAULT_VIDEO_SECONDS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "dashboard",
        help="Directory for captured media. Default: presentation/output/dashboard",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the intended capture plan without starting anything.",
    )
    parser.add_argument("--verbose", action="store_true", help="Emit DEBUG-level log output.")
    parser.add_argument(
        "--version", action="version", version=f"capture_dashboard_media.py {__version__}"
    )
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> None:
    # WHAT: Range-checks the port and video duration, prepares the output dir.
    # WHY:  An out-of-range port produces a confusing bind error deep inside Dash.
    # HOW:  Explicit bounds checks raising ValueError for exit code 2.
    if not 1 <= int(args.port) <= 65535:
        raise ValueError(f"--port must be in 1..65535, got {args.port}")
    if float(args.video_seconds) <= 0.0:
        raise ValueError("--video-seconds must be greater than 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def start_dashboard(host: str, port: int) -> subprocess.Popen:
    # WHAT: Launches scripts/run_dashboard.py as a child process.
    # WHY:  Reusing the existing entrypoint guarantees the captured dashboard is
    #       the one the project actually ships, not a bespoke variant.
    # HOW:  subprocess.Popen on the current interpreter with the repo as cwd.
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_dashboard.py"),
        "--host", host,
        "--port", str(port),
    ]
    return subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_until_reachable(url: str, timeout_s: float) -> bool:
    # WHAT: Polls the dashboard URL until it answers or the budget expires.
    # WHY:  Navigating before Dash has bound its port yields a misleading
    #       ERR_CONNECTION_REFUSED that looks like a code fault rather than a race.
    # HOW:  Bounded polling loop - no unbounded wait, per the JPL bounded-loop rule.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - fixed localhost URL
                if 200 <= response.status < 400:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(STARTUP_POLL_S)
    return False


def capture(args: argparse.Namespace, url: str) -> int:
    # WHAT: Drives Playwright over the live dashboard, writing PNGs and video.
    # WHY:  Separating capture from process lifecycle keeps main() a call sequence.
    # HOW:  One context with optional record_video_dir; a scripted scroll pass so
    #       the video shows the full page rather than a static viewport.
    from playwright.sync_api import sync_playwright

    logger = configure_logging(args.verbose)
    staging = args.output_dir / "_capture"
    if args.record_video:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

    context_options: dict[str, Any] = {
        "viewport": {"width": SLIDE_WIDTH_PX, "height": SLIDE_HEIGHT_PX},
    }
    if args.record_video:
        context_options["record_video_dir"] = str(staging)
        context_options["record_video_size"] = {
            "width": SLIDE_WIDTH_PX,
            "height": SLIDE_HEIGHT_PX,
        }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--force-color-profile=srgb"])
        context = browser.new_context(**context_options)
        try:
            page = context.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(RENDER_SETTLE_MS)

            page.screenshot(path=str(args.output_dir / "dashboard_viewport.png"))
            page.screenshot(path=str(args.output_dir / "dashboard_full.png"), full_page=True)
            logger.info("Screenshots written to %s", args.output_dir)

            if args.record_video:
                steps = 12
                per_step_ms = max(1, int(float(args.video_seconds) * 1000 / steps))
                for step in range(steps):
                    fraction = step / float(steps - 1) if steps > 1 else 0.0
                    page.evaluate(
                        "(f) => window.scrollTo({top: f * (document.body.scrollHeight - window.innerHeight),"
                        " behavior: 'smooth'})",
                        fraction,
                    )
                    page.wait_for_timeout(per_step_ms)
        finally:
            context.close()
            browser.close()

    if args.record_video:
        captures = sorted(staging.glob("*.webm"))
        if not captures:
            raise RuntimeError(f"Playwright produced no video file in {staging}")
        target = args.output_dir / "xquantx_dashboard.webm"
        target.unlink(missing_ok=True)
        captures[0].replace(target)
        shutil.rmtree(staging, ignore_errors=True)
        logger.info(
            "Video written: %s (%.1f MB)", target, target.stat().st_size / (1024 * 1024)
        )
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(args.verbose)
    url = f"http://{args.host}:{args.port}/"
    try:
        validate_inputs(args)
    except (ValueError, OSError) as exc:
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
        logger.info("Dry run: would capture %s into %s", url, args.output_dir)
        return EXIT_DRY_RUN

    process: Optional[subprocess.Popen] = None
    try:
        if not args.attach:
            logger.info("Starting dashboard at %s", url)
            process = start_dashboard(args.host, args.port)
        if not wait_until_reachable(url, STARTUP_TIMEOUT_S):
            logger.error("Dashboard did not become reachable at %s within %ss", url, STARTUP_TIMEOUT_S)
            return EXIT_SERVICE_UNREACHABLE
        return capture(args, url)
    except Exception as exc:  # noqa: BLE001 - top-level boundary, logged then coded
        logger.error("Capture failed: %s", exc)
        return EXIT_FAILURE
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
