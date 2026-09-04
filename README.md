# xquantx

> **Current project notice (2026-08-28).** The active product is **X Quant X**, a
> quantitative paper-trading platform. The earlier voice-compliance description below
> is retained as historical baseline content; the current product definition is
> `alpaca_paper_trading_specifications_x_quant_x/001_xquantx_concept.txt`.

## Current Status

X Quant X uses Alpaca's paper-trading API to transform market data into risk-gated
paper orders. MVP scope includes a P&L/risk dashboard, optional backtesting, and
options paper trading; live-money trading is out of scope.

**Company:** Not yet confirmed (open item) &nbsp;|&nbsp; **Category:** Voice Compliance QA (VCQ, proposed) &nbsp;|&nbsp; **Status:** Hackathon-stage development

An automated compliance and red-team test bench for voice AI agents: simulates callers across
a range of personas and adversarial scenarios, runs each through a five-phase call lifecycle,
transcribes both sides of the conversation, and scores the target agent against statutory
requirements (verification-flagged) and the target company's own published terms, producing a
Pass/Fail/Flag verdict with an explicit rule citation and an advisory-only suggested
system-prompt patch on failure.

Intended for an assumed **LabLab AI Factory** hackathon submission using **Natively AI** (window
inherited from a sibling project, not yet independently confirmed). This repository
(`X_Voice_X`, a `py.typed` Python package) is the authoritative backend implementation target,
not a disposable planning artifact.

## Table of Contents

- [xquantx](#xquantx)
  - [Current Status](#current-status)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Presentation and Submission Material](#presentation-and-submission-material)
  - [Documentation](#documentation)
  - [Getting Started](#getting-started)
  - [Project Status](#project-status)
  - [Contributing](#contributing)
  - [Security](#security)
  - [License](#license)
  - [Reconciliation](#reconciliation)
  - [Changelog](#changelog)

## About

See [ABOUT.md](ABOUT.md) for the full product identity, elevator pitch, and project origin.

## Presentation and Submission Material

The LabLab AI submission set lives in [`presentation/`](presentation/):

- [`presentation/DESCRIPTION_SHORT.md`](presentation/DESCRIPTION_SHORT.md) — submission copy at four lengths, plus a verified-facts table
- [`presentation/DESCRIPTION_LONG.md`](presentation/DESCRIPTION_LONG.md) — full narrative with architecture, Kalman, sequence, and capital-gate diagrams
- [`presentation/README.md`](presentation/README.md) — how the PDF deck, per-slide images, and the full-HD walkthrough video are generated from one HTML source by the Playwright scripts in [`scripts/presentation/`](scripts/presentation/)

## Documentation

All product, design, and engineering specifications live in
[`alpaca_paper_trading_specifications_x_quant_x/`](alpaca_paper_trading_specifications_x_quant_x/), governed by a binding
documentation standard (`0000_documentation_standards.txt`). Start at the master index:

- [`alpaca_paper_trading_specifications_x_quant_x/000_index.txt`](alpaca_paper_trading_specifications_x_quant_x/000_index.txt) — full document map
- [`alpaca_paper_trading_specifications_x_quant_x/001_voice_transcription_concept.txt`](alpaca_paper_trading_specifications_x_quant_x/001_voice_transcription_concept.txt) — product identity, concept, MVP scope, open items
- [`alpaca_paper_trading_specifications_x_quant_x/017_xquantx_scoring_rules.txt`](alpaca_paper_trading_specifications_x_quant_x/017_xquantx_scoring_rules.txt) — authoritative scoring/verdict engine

## Getting Started

1. Set up the Python virtual environment per [`alpaca_paper_trading_specifications_x_quant_x/007_xquantx_virtual_env.txt`](alpaca_paper_trading_specifications_x_quant_x/007_xquantx_virtual_env.txt).
2. Review the repository scaffolding in [`alpaca_paper_trading_specifications_x_quant_x/006_xquantx_scaffolding.txt`](alpaca_paper_trading_specifications_x_quant_x/006_xquantx_scaffolding.txt).
3. Review coding standards in [`alpaca_paper_trading_specifications_x_quant_x/005_xquantx_coding_standards.txt`](alpaca_paper_trading_specifications_x_quant_x/005_xquantx_coding_standards.txt) before opening a Pull Request.

## Project Status

Active hackathon-stage development (assumed LabLab AI Factory / Natively AI, window not yet
confirmed). No versioned release has shipped yet; see [SECURITY.md](SECURITY.md) for the
supported-version policy and [`alpaca_paper_trading_specifications_x_quant_x/015_xquantx_deployment.txt`](alpaca_paper_trading_specifications_x_quant_x/015_xquantx_deployment.txt) for the deployment plan.

## Contributing

Contributions are welcome via **mandatory fork-and-Pull-Request** (direct pushes to upstream are not accepted). See [CONTRIBUTION.md](CONTRIBUTION.md) for the full workflow, coding standards, and testing requirements.

## Security

To report a vulnerability, do not open a public issue — see [SECURITY.md](SECURITY.md) for the private reporting process.

## License

See [LICENSE.md](LICENSE.md) for full license terms.

## Reconciliation

| Earlier wording | Current canon | Source |
| --- | --- | --- |
| Voice Compliance QA test bench | Quantitative paper-trading and market-intelligence platform | `001_xquantx_concept.txt` Section 1.3 |
| Natively AI hackathon dependency | Alpaca paper-trading API dependency | `001_xquantx_concept.txt` Section 1.4.3 |
| Voice-agent transcript scoring | Market regime inference, risk gating, and paper-order execution | `001_xquantx_concept.txt` Section 1.7 |

## Changelog

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 2026.9.3.1 | 2026-09-03 | Hadrian Hu | Added the Presentation and Submission Material section linking the LabLab deck, descriptions, and Playwright media pipeline. |
| 2026.8.28.1 | 2026-08-28 | GitHub Copilot | Added current project facts and reconciled retained voice-era content with the X Quant X canon. |
