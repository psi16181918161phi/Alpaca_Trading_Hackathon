"""Live probe for the four Featherless accounts.

Run this with the real keys in the environment to confirm each model
responds. It writes a small smoke-test record to ``llm_usage.jsonl`` so
you can see actual latency and token counts.

Usage:
    export FEATHERLESS_DEEPHERMES_KEY=rc_...
    export FEATHERLESS_FINANCE_LLAMA_KEY=rc_...
    export FEATHERLESS_QWEN_TRADING_KEY=rc_...
    export FEATHERLESS_RESERVE_KEY=rc_...
    python scripts/probe_featherless.py

The script DOES NOT mutate any state. It just calls each provider once
with a tiny prompt and prints the result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make src/ importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_dotenv()

from investment_agent.llm import (
    FeatherlessOrchestrator,
    ProviderSpec,
    UsageLog,
    build_named_specialists,
    build_snapshot,
    pre_screen,
    run_named_specialists,
)


def main() -> int:
    required = [
        "FEATHERLESS_DEEPHERMES_KEY",
        "FEATHERLESS_FINANCE_LLAMA_KEY",
        "FEATHERLESS_QWEN_TRADING_KEY",
        "FEATHERLESS_RESERVE_KEY",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print("ERROR: missing env vars:", ", ".join(missing))
        return 2

    usage = UsageLog(log_file="llm_usage.jsonl")

    specs = [
        ProviderSpec("deephermes", os.getenv("FEATHERLESS_DEEPHERMES_KEY"),
                     "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                     0.22, 850, "reasoning", is_reserve=False),
        ProviderSpec("fundamentals", os.getenv("FEATHERLESS_FUNDAMENTALS_KEY"),
                     "NousResearch/DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos",
                     0.15, 700, "fundamental", is_reserve=False),
        ProviderSpec("finance_qlora", os.getenv("FEATHERLESS_FINANCE_LLAMA_KEY"),
                     "jhon53/Llama3_1_8B_Finance_QLoRA-merged-16bit",
                     0.12, 700, "fundamental", is_reserve=False),
        ProviderSpec("reserve", os.getenv("FEATHERLESS_RESERVE_KEY"),
                     "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                     0.15, 700, "failover", is_reserve=True),
    ]

    orch = FeatherlessOrchestrator(specs, usage_log=usage, retries_per_provider=2)
    print(f"Active providers: {orch.active_provider_ids}")
    print(f"Reserve providers: {orch.reserve_provider_ids}")

    snapshot = build_snapshot("SPY", [100 + 0.1 * i for i in range(20)], regime="R01")
    screen = pre_screen("SPY", [100 + 0.1 * i for i in range(20)], previous_snapshot=None)
    print(f"Pre-screen: should_call_llm={screen.should_call_llm}, reason={screen.reason}")

    print("\n--- direct ping per primary provider ---")
    for pid in orch.active_provider_ids + orch.reserve_provider_ids:
        try:
            response = orch.complete(
                'Return exactly this JSON: {"signal": 0.0, "confidence": 0.5, "uncertainty": 0.5, "doubt": 0.5, "p_plus": 0.5, "p_minus": 0.5, "delta_t": 1.0, "noise": 0.1}',
                provider_id=pid,
                max_tokens=64,
            )
            print(f"[OK]   {pid}: {response.text[:120]!r} "
                  f"({response.prompt_tokens}+{response.completion_tokens} tokens, "
                  f"{response.latency_ms:.0f}ms)")
        except Exception as exc:
            print(f"[FAIL] {pid}: {exc!r}")

    print("\n--- named specialists end-to-end ---")
    specialists = build_named_specialists(orch)
    outputs = run_named_specialists(specialists, snapshot)
    for aid, out in outputs.items():
        print(f"  {aid}: s={out.s:+.3f} c={out.c:.3f} u={out.u:.3f} d={out.d:.3f}")

    total = usage.total_tokens()
    print(f"\nTotal tokens spent during this probe: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
