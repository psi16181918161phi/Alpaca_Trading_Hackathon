---
title: "X Quant X — Presentation and Submission Media Pipeline"
author: "Hadrian Hu"
date: "2026-09-03"
version: "2026.1.0.0"
keywords: ["deck", "pdf", "playwright", "presentation", "submission", "video"]
status: "Active"
---

# X Quant X — Presentation and Submission Media Pipeline

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Keywords](#2-keywords)
- [3. Executive Summary](#3-executive-summary)
- [4. Directory Contents](#4-directory-contents)
- [5. Prerequisites](#5-prerequisites)
- [6. Build Commands](#6-build-commands)
- [7. Design System](#7-design-system)
- [8. Why Playwright and Not a Slide Tool](#8-why-playwright-and-not-a-slide-tool)
- [9. Exit Codes](#9-exit-codes)
- [10. Editing the Deck](#10-editing-the-deck)
- [11. References](#11-references)
- [Changelog](#changelog)

---

## 1. Abstract

This directory holds the LabLab AI submission material for X Quant X: the
short-form and long-form project descriptions, the HTML source of the
presentation deck, and the Playwright-driven scripts that render that single
source into a paginated full-HD PDF, a per-slide PNG set, a long-form 1920x1080
walkthrough video, and captured media of the live control-room dashboard.

## 2. Keywords

deck, pdf, playwright, presentation, submission, video, xquantx

## 3. Executive Summary

One HTML deck is the single source of truth. The PDF, the PNG set and the video
are all generated from it by script, so they cannot drift apart. Nothing here is
hand-assembled in a slide editor, and every generated artefact is reproducible
from a clean checkout with two commands.

## 4. Directory Contents

| Path | Description |
| :--- | :--- |
| `DESCRIPTION_SHORT.md` | Ready-to-paste submission copy at four lengths, plus a verified-facts table. |
| `DESCRIPTION_LONG.md` | Full submission narrative with Mermaid architecture, Kalman, sequence and capital-gate diagrams. |
| `deck/index.html` | The deck itself: 16 slides, inline SVG diagrams, no runtime JS dependencies beyond `deck.js`. |
| `deck/theme.css` | Palette, typography, spacing and print tokens. The only place a hex literal may appear. |
| `deck/deck.js` | Slide navigation, exposing the `window.XQX` API the scripts drive. |
| `output/` | Generated artefacts. Not authored by hand; safe to delete and rebuild. |

Generation scripts live in [`scripts/presentation/`](../scripts/presentation),
not here, so the media pipeline follows the same scripting conventions as every
other executable in the repository.

## 5. Prerequisites

```powershell
python -m pip install playwright imageio-ffmpeg
python -m playwright install chromium
```

`imageio-ffmpeg` is required only for the `--mp4` transcode; the WebM capture
works without it. Both are already listed in `requirements-dev.txt`.

## 6. Build Commands

### 6.1. PDF deck plus per-slide PNGs

```powershell
python scripts/presentation/render_deck_pdf.py --png-dir presentation/output/slides
```

Produces `presentation/output/xquantx_deck.pdf` (one 1920x1080 page per slide,
backgrounds printed) and `slide_01.png` ... `slide_16.png` at a device scale
factor of 2.

### 6.2. Long-form HD walkthrough video

```powershell
python scripts/presentation/record_deck_video.py --seconds-per-slide 16 --intro-seconds 5 --outro-seconds 8 --mp4
```

Produces `presentation/output/xquantx_deck.webm` and, with `--mp4`,
`xquantx_deck.mp4` encoded H.264 / yuv420p at CRF 18 for text sharpness.
Recording happens in real time: the wall-clock duration of the command is
approximately the duration of the resulting video.

### 6.3. Live dashboard capture

```powershell
python scripts/presentation/capture_dashboard_media.py --record-video
```

Starts `scripts/run_dashboard.py` in a child process, waits for the HTTP endpoint
to answer, captures a viewport screenshot, a full-page screenshot and an optional
scrolling walkthrough video into `presentation/output/dashboard/`, then
terminates the child process it started. Pass `--attach` to capture a dashboard
you started yourself; in that mode the script never terminates the server.

### 6.4. Rehearsing without producing anything

Every script accepts `--dry-run`, which validates inputs and reports the intended
actions without launching a browser, and `--verbose` for DEBUG logging.

## 7. Design System

The deck palette mirrors `src/investment_agent/dashboard/colors.py` token for
token, so the deck, the PDF, the video and the running dashboard are visibly the
same product.

| Token | Value | Role |
| :--- | :--- | :--- |
| `--bg-primary` | `#000000` | Page background |
| `--bg-card` | `#170f11` | Data cards and chart panels |
| `--accent` | `#B76E79` | Rose gold: primary accent, rules, series-1 |
| `--alert` | `#FFAEC9` | Variant B: alert cards and emphasis, always with black text |
| `--text-primary` | `#f2ecec` | Body and heading text |
| `--positive` | `#2DD4BF` | Positive values, ALLOW verdict |
| `--alert-badge` | `#e11d48` | Compact critical badge, BLOCK verdict |

Typography is Inter for prose and JetBrains Mono for every numeric value,
matching the dashboard. `--alert` is reserved: it marks the single most
load-bearing statement on a slide, never decoration.

## 8. Why Playwright and Not a Slide Tool

8.1. **One source, many artefacts.** A PowerPoint file and an exported video are
two artefacts that drift. An HTML deck plus two scripts cannot.

8.2. **The diagrams are the product.** Inline SVG uses the same palette tokens as
the dashboard, so a palette change propagates to every diagram automatically.

8.3. **Reproducibility.** The deck rebuilds byte-for-similar from a clean
checkout with no manual steps, which is the same standard the rest of the
repository is held to.

8.4. **Offline determinism.** The deck depends on no CDN-hosted diagram renderer.
Google Fonts is requested but every family has a local fallback stack, so a
disconnected build still renders correctly.

## 9. Exit Codes

All three scripts follow `python_scripting_standards.txt` §7.2:

| Code | Meaning |
| :--- | :--- |
| 0 | Success |
| 1 | General failure |
| 2 | Invalid arguments or missing input file |
| 3 | Missing dependency (Playwright not installed) |
| 4 | Dry run complete, no changes made |
| 20 | External service error (dashboard never became reachable) |

## 10. Editing the Deck

10.1. Add a slide by appending a `<section class="slide">` to `deck/index.html`.
`deck.js` renumbers every slide automatically, and both scripts read the count
from `window.XQX.count()` — no constant anywhere needs updating.

10.2. Never write a hex literal outside the `:root` block in `theme.css`. The two
`<marker>` fills in `index.html` are the sole documented exception, because SVG
markers cannot resolve CSS custom properties in the `fill` attribute.

10.3. Re-run both build commands after any edit. A stale PDF beside a fresh deck
is exactly the drift this pipeline exists to prevent.

## 11. References

11.1. `coding_stds/CodingStandardsRef/coding_stds/scripting/python_scripting_standards.txt` — script anatomy, argparse, logging, exit codes, type annotations.

11.2. `coding_stds/CodingStandardsRef/coding_stds/scripting/scripting_standards_workflows_pipelines.txt` — script classification and pipeline composition.

11.3. `coding_stds/CodingStandardsRef/coding_stds/visualization/aesthetic_standards.txt` — palette single-source-of-truth rule, typography and spacing tokens.

11.4. `coding_stds/CodingStandardsRef/coding_stds/documentation/markdown_standards.txt` — this document's structure.

11.5. `src/investment_agent/dashboard/colors.py` — the palette this deck mirrors.

## Changelog

| Version | Date | Author | Description |
| :--- | :--- | :--- | :--- |
| 2026.1.0.0 | 2026-09-03 | Hadrian Hu | Initial creation. Documents the deck source, the three Playwright generation scripts, the design-system token table, exit-code contract, and the editing rules that keep PDF, PNG and video in sync. |
