# Specialist Agent Prompts

> This file is the **authoritative reference** for the structured prompts the
> multi-agent layer (`src/investment_agent/agents/specialist.py`) sends to
> the LLM provider. Every prompt must require the eight-channel JSON
> response so the downstream pipeline (ensemble → Kalman → capital gate)
> never has to deal with free-form prose.

The seven default roles cover the four financial domains of the public
whitepaper plus the cross-market dimension:

| # | agent_id            | domain | role                                         |
|---|---------------------|--------|----------------------------------------------|
| 1 | `agent_economic`    | earth  | Economic state specialist                    |
| 2 | `agent_financial`   | earth  | Financial-plumbing specialist                |
| 3 | `agent_fiscal`      | earth  | Fiscal / policy specialist                   |
| 4 | `agent_portfolio`   | earth  | Portfolio risk specialist                    |
| 5 | `agent_fundamental` | earth  | Earnings / valuation specialist              |
| 6 | `agent_market`      | water  | Microstructure / liquidity specialist        |
| 7 | `agent_sector`      | water  | Cross-market / sector rotation specialist    |

A separate `agent_volatility` (Fire domain) is available for options-aware
extensions.

---

## Universal JSON contract

Every agent must emit a **single JSON object** inside a ` ```json` fenced
block. The keys and domains are:

| key           | domain      | meaning                                |
|---------------|-------------|----------------------------------------|
| `signal`      | `[-1, 1]`   | Directional conviction                 |
| `confidence`  | `(0, 1]`    | How sure the agent is                  |
| `uncertainty` | `[0, 1]`    | Internal conflict / information gap    |
| `doubt`       | `[0, 1]`    | Calibration quality (0 = excellent)    |
| `p_plus`      | `[0, 1]`    | Probability of a favourable outcome    |
| `p_minus`     | `[0, 1]`    | Probability of an unfavourable outcome |
| `delta_t`     | `> 0`       | Time horizon in bars                   |
| `noise`       | `> 0`       | Self-estimated measurement noise       |

Anything outside the JSON block is ignored. If the LLM returns a
malformed object, the adapter substitutes a **zero signal, low-confidence**
fallback so the deterministic risk layer is never starved.

---

## Common user-prompt template

Each agent receives a structured user prompt assembled from:

- `symbol` (e.g., `AAPL`)
- `regime` (e.g., `R01`) and the regime posterior probability vector
- `features` (regime feature summary: RSI, MACD, ATR, VIX, etc.)
- current `ensemble_signal` and `disagreement`
- `memory` (top-5 similar past closed trades, with similarity score)
- `peer_agents` (the other agents' most recent signals)

This is rendered by `SpecialistAgent._build_prompt` and is the same shape
across all seven roles — only the system prompt differs.

---

## Per-agent system prompts

The current system prompts live in `src/investment_agent/agents/specialist.py`
(``ECONOMIC_ROLE``, ``FINANCIAL_ROLE``, …, ``SECTOR_ROLE``). Each one:

1. Anchors the agent to a single financial domain.
2. Caps the absolute signal magnitude to keep any one LLM from dominating
   the ensemble (e.g., the fundamental specialist is capped at 0.5).
3. Requires the JSON contract above.

These prompts are intentionally **short** so the LLM spends its budget on
the data, not the persona.
