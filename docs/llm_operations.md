# LLM Operations Guide

This file explains how to run the X Quant X agent against the three
Featherless models from the spec.

## 1. Set up the API keys

Set these in your shell (or `.env`, or `config/llm_keys.json`):

```bash
export FEATHERLESS_DEEPHERMES_KEY=rc_6036b7e2...
export FEATHERLESS_FINANCE_LLAMA_KEY=rc_9a5c48a9...
export FEATHERLESS_QWEN_TRADING_KEY=rc_be4ce070...
export FEATHERLESS_RESERVE_KEY=rc_fb3794da...   # failover only
```

`config/llm_keys.json` is gitignored. A template lives at
`config/llm_keys.json.example`.

## 2. The four-provider model

| Slot      | Model                                                    | Temp   | Max tokens | Role             |
|-----------|----------------------------------------------------------|--------|------------|------------------|
| primary 1 | `NousResearch/DeepHermes-3-Llama-3-8B-Preview`           | 0.22   | 850        | reasoning        |
| primary 2 | `instruction-pretrain/finance-Llama3-8B`                 | 0.12   | 600        | fundamental      |
| primary 3 | `precisionalgorithms/qwen3.5-9b_precision_agentic_trading` | 0.15 | 750        | execution-context|
| reserve   | `NousResearch/DeepHermes-3-Llama-3-8B-Preview` (or any)  | 0.15   | 700        | failover only    |

Each specialist has a hard cap on absolute signal magnitude:

- DeepHermes — 0.5
- Finance-Llama — 0.4
- Qwen Trading — 0.6

The reserve is *only* consulted if all three primaries fail, so a single
bad key does not blow the budget.

## 3. Running the agent

```python
import sys
sys.path.insert(0, "src")

from investment_agent.llm import (
    load_provider_specs, FeatherlessOrchestrator, UsageLog,
    build_named_specialists, run_named_specialists,
    build_snapshot, pre_screen,
)
from investment_agent.orchestrator import XQuantXOrchestrator

specs = load_provider_specs()
orchestrator = FeatherlessOrchestrator(specs, usage_log=UsageLog())
specialists = build_named_specialists(orchestrator)

prices = [...]  # most recent last
snapshot = build_snapshot("SPY", prices, regime="R02")
decision = pre_screen("SPY", prices, previous_snapshot=None)
if not decision.should_call_llm:
    # Skip LLM cost: ensemble already has a confident enough read.
    pass
else:
    outputs = run_named_specialists(specialists, snapshot)
    # outputs: {agent_id: AgentOutput}  ->  hand to the existing ensemble
```

## 4. Tracking the $25 budget

Every successful and failed LLM call is appended to `llm_usage.jsonl`:

```json
{"timestamp": "2026-09-01T10:21:33.451", "provider_id": "deephermes", "model": "NousResearch/DeepHermes-3-Llama-3-8B-Preview", "success": true, "latency_ms": 812.0, "prompt_tokens": 420, "completion_tokens": 220, "error": ""}
```

Use `UsageLog.total_tokens()` to read the cumulative token count from disk.

## 5. Failure modes

- **Provider timeout** → retry once, then fail over to the next active
  provider, then to the reserve.
- **Malformed JSON** → `AgentLLMAdapter` falls back to a zero-signal,
  low-confidence `AgentOutput`. The risk gate still runs.
- **No API key** → that provider is skipped silently (the orchestrator
  tries the next one).
- **All providers fail** → orchestrator raises `RuntimeError`. The
  pipeline should catch this and HOLD.

## 6. What the LLM is NOT allowed to do

The LLM can only contribute to `(s, c, u, d, p_plus, p_minus, delta_t, r)`
for each specialist. It cannot:

- modify the capital gate verdict
- bypass the liquidity floor
- change the SoC state
- ignore risk flags (Qwen explicitly drops confidence below 0.4 when
  `risk_flags` is non-empty)
- set its own position size
- bypass the circuit breakers

The deterministic pipeline remains the hard authority for every decision.
