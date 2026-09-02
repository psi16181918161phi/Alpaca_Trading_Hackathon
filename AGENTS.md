# X QUANT X — Autonomous Trading System & Agent Guidelines

## System Overview

`X QUANT X` is an autonomous quantitative trading engine integrating 7 specialized LLM agents, an HMM-based regime detector, a Kalman filter signal smoother, a 7-State Capital Gate risk model, and an Alpaca execution layer.

---

## 7 Specialist Agents

1. **Economic State Specialist** (`FEATHERLESS_DEEPHERMES_KEY`)
   - Model: `nousresearch/hermes-3-llama-3.8b`
   - Analyzes macro indicators, inflation, interest rate expectations, and GDP trends.

2. **Financial State Specialist** (`FEATHERLESS_FINANCE_LLAMA_KEY`)
   - Model: `finance-llama-8b` / `meta-llama/Meta-Llama-3.1-8B-Instruct`
   - Evaluates liquidity, credit spreads, yield curve shapes, and systemic stability.

3. **Fiscal State Specialist** (`FEATHERLESS_DEEPHERMES_KEY`)
   - Model: `nousresearch/hermes-3-llama-3.8b`
   - Monitors government spending, debt issuance, and taxation policy impacts.

4. **Portfolio State Specialist** (`FEATHERLESS_DEEPHERMES_KEY`)
   - Model: `nousresearch/hermes-3-llama-3.8b`
   - Assesses current exposure, drawdown limits, leverage, and diversification.

5. **Fundamental State Specialist** (`FEATHERLESS_FINANCE_LLAMA_KEY`)
   - Model: `finance-llama-8b`
   - Analyzes equity earnings metrics, valuation ratios, balance sheets, and growth.

6. **Market State Specialist** (`FEATHERLESS_QWEN_TRADING_KEY` / `FEATHERLESS_DEEPHERMES_KEY`)
   - Model: `Qwen/Qwen2.5-72B-Instruct` / `nousresearch/hermes-3-llama-3.8b`
   - Analyzes order book dynamics, volume profile, volatility (ATR/VIX proxy), and momentum.

7. **Sector Specialist** (`FEATHERLESS_DEEPHERMES_KEY`)
   - Model: `nousresearch/hermes-3-llama-3.8b`
   - Evaluates sector relative strength, industry tailwinds, and cross-asset correlations.

---

## Environment & Provider Configuration

Configure API credentials in `.env` or `config/llm_keys.json`:

```bash
# LLM Providers (Featherless)
FEATHERLESS_DEEPHERMES_KEY=your_deephermes_key
FEATHERLESS_FINANCE_LLAMA_KEY=your_finance_llama_key
FEATHERLESS_RESERVE_KEY=your_reserve_key

# Broker Credentials (Alpaca Paper Trading)
APCA_API_KEY_ID=your_alpaca_key_id
APCA_API_SECRET_KEY=your_alpaca_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

> **Note**: For production/paper live trading (`--stage paper` or `--stage competition`), real Featherless credentials are **mandatory**. Silent fallback to `MockLLMProvider` is disabled in live modes to ensure authentic model execution.

---

## Autonomous Execution Lifecycle

```
 Market Data (Prices/Volumes)
         │
         ▼
 Market Feature Extractor (RSI, ATR, VIX) & HMM Regime Classifier (R01-R12)
         │
         ▼
 Trade Memory Retrieval (Top-K Similar Historical Experiences)
         │
         ▼
 7 LLM Specialist Agents (Ensemble Signal & Disagreement Computation)
         │
         ▼
 Agent Reputation Weighting (Beta Binomial Tracker - Restored from State)
         │
         ▼
 Kalman Filter Signal Smoother & Investment Gain
         │
         ▼
 7-State Capital Gate Risk Sanity Check
         │
         ▼
 Product Gate Selection (Equity vs Options vs Cash)
         │
         ▼
 Alpaca Execution Engine & Emergency Circuit Breaker
         │
         ▼
 Reconciliation & Closed Learning Loop (Reputation & Memory Update)
```

---

## Command Reference

### 1. Persistent Autonomous Live Loop
Run the continuous live loop with candidate screening and market data streaming:

```bash
python scripts/run_live_loop.py --stage dry_run --interval 60 --top-n 3
```

For live paper trading with broker submission:
```bash
python scripts/run_live_loop.py --stage paper --interval 300 --top-n 5
```

### 2. Single Candidate Paper Loop
Run the comprehensive paper loop on a single ticker:

```bash
python scripts/run_paper_loop.py --symbol AAPL --reputation reputation_state.json --memory trade_memory.json
```

### 3. Dashboard Control Interface
Launch the interactive web dashboard with session control and manual order placement:

```bash
python src/investment_agent/dashboard/app.py
```
