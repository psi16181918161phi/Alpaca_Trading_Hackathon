---
title: "X Quant X — Short-Form Project Description (LabLab AI Submission)"
author: "Hadrian Hu"
date: "2026-09-03"
version: "2026.1.0.0"
keywords: ["description", "lablab", "short-form", "submission", "xquantx"]
status: "Active"
---

# X Quant X — Short-Form Project Description

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Keywords](#2-keywords)
- [3. Executive Summary](#3-executive-summary)
- [4. Submission Copy Blocks](#4-submission-copy-blocks)
- [5. References](#5-references)
- [Changelog](#changelog)

---

## 1. Abstract

This document holds the ready-to-paste short-form copy for the LabLab AI
submission form. Every claim here is traceable to
`alpaca_paper_trading_specifications_x_quant_x/001_xquantx_concept.txt` or to
the verified test/report artefacts cited in §5. Nothing in this file is an
estimate, a projection, or an unverified performance claim.

## 2. Keywords

alpaca, lablab, multi-agent, paper-trading, quantitative-finance, regime-detection,
risk-gate, submission-copy

## 3. Executive Summary

X Quant X is a multi-agent quantitative **paper-trading** platform built on the
Alpaca paper-trading API. It infers the prevailing market regime, fuses seven
independent LLM specialist opinions into a single reputation-weighted signal,
smooths that signal with a Kalman filter, and forces every resulting order
through a seven-state capital gate before a single share is submitted. Live-money
trading is explicitly out of scope.

## 4. Submission Copy Blocks

### 4.1. Tagline (1 line, 62 characters)

> Predict less blindly. Measure more completely. Control risk before capital.

### 4.2. One-sentence description (198 characters)

> X Quant X is a multi-agent quantitative paper-trading platform that infers the
> market regime, fuses seven specialist LLM signals into one risk-gated decision,
> and executes it through Alpaca.

### 4.3. Short description (~60 words)

> X Quant X turns market data into risk-gated paper trades. An HMM classifies the
> current regime across twelve archetypes; seven specialist LLM agents each score
> the setup from a different lens; a reputation tracker weights them by their own
> track record; a Kalman filter smooths the fused signal; and a seven-state capital
> gate must approve before Alpaca receives an order.

### 4.4. Medium description (~120 words)

> X Quant X is an autonomous quantitative paper-trading engine. Every cycle begins
> with feature extraction and an HMM regime classification over twelve market
> archetypes (R01–R12). Seven specialist LLM agents — Economic, Financial, Fiscal,
> Portfolio, Fundamental, Market, and Sector — independently score the opportunity.
> A Beta-Binomial reputation tracker weights each agent by its own realised hit
> rate, and their disagreement is measured rather than hidden. The fused signal is
> smoothed by a two-dimensional Kalman filter, then submitted to a seven-state
> capital gate that can ALLOW, REDUCE, BLOCK, or FLATTEN. Only surviving decisions
> reach the Alpaca paper API. Outcomes are reconciled back into agent reputation
> and trade memory, closing the learning loop.

### 4.5. Category and stack

- **Category:** Quantitative Finance / Algorithmic Paper Trading / Market Intelligence
- **Primary API:** Alpaca paper-trading API (`alpaca-py`)
- **Languages:** Python (primary), C++ (numerical kernel), Go, Rust
- **Interface:** Dash/Plotly control-room dashboard (equity, drawdown, regime
  probability, risk-gate audit trail, order blotter)

### 4.6. Verified engineering facts

These are the only quantitative claims cleared for submission use, each with a
recorded source:

| Claim | Value | Source |
| :--- | :--- | :--- |
| Automated test suite | 868 passed, 14 subtests, 0 failed | `CHATS/2026-09-02_...-REPORT.md` §10 |
| Statement coverage | 82% (6419 statements) | `CHATS/2026-09-02_...-REPORT.md` §10 |
| C++ Kalman kernel parity | 11/11 GoogleTest, exact double-precision match with Python | `CHATS/2026-09-02_...-REPORT.md` §8 |
| Live-money trading | Out of scope | `001_xquantx_concept.txt` §1.9.2 |

**Not claimed:** returns, Sharpe ratio, win rate, or any performance figure. No
such figure has been independently verified, so none appears in any submission
artefact.

## 5. References

5.1. `alpaca_paper_trading_specifications_x_quant_x/001_xquantx_concept.txt` — canonical concept, brand, MVP scope.

5.2. `AGENTS.md` — the seven specialist agents and the execution lifecycle.

5.3. `CHATS/2026-09-02_archive-consolidation-regression-suite-REPORT.md` — verified test, coverage, and C++ parity figures.

5.4. `coding_stds/CodingStandardsRef/coding_stds/documentation/markdown_standards.txt` — governing document structure.

## Changelog

| Version | Date | Author | Description |
| :--- | :--- | :--- | :--- |
| 2026.1.0.0 | 2026-09-03 | Hadrian Hu | Initial creation. Short-form LabLab submission copy at four lengths, category/stack block, and a verified-facts table with an explicit no-performance-claims rule. |
