"""LLM package — providers, failover, and adapters for the X Quant X agent layer.

Public API:
    LLMProvider            — abstract protocol
    LLMResponse            — typed result of a single LLM call
    MockLLMProvider        — deterministic in-process provider (offline / tests)
    FeatherlessProvider    — Featherless chat-completions HTTP backend
    FeatherlessOrchestrator — multi-key Featherless orchestrator with reserve failover
    UsageLog               — JSONL usage logger for the $25 budget tracker
    AgentLLMAdapter        — bridges LLM output → ``AgentOutput``
    extract_json_object    — defensive JSON extractor
    build_snapshot         — compact LLM state builder
    pre_screen             — deterministic pre-screen for skipping LLM calls
    DEEPHERMES_ROLE / FINANCE_LLAMA_ROLE / QWEN_TRADING_ROLE / NAMED_ROLES
    NamedSpecialist / build_named_specialists / run_named_specialists
"""

from .base import (
    LLMProvider,
    LLMResponse,
    MockLLMProvider,
    FeatherlessProvider,
)
from .adapter import AgentLLMAdapter, extract_json_object
from .orchestrator import (
    FeatherlessOrchestrator,
    FailureKind,
    ProviderSpec,
    UsageLog,
    UsageRecord,
    classify_failure,
    load_provider_specs,
)
from .snapshot import build_snapshot, pre_screen, PreScreenResult
from .named import (
    DEEPHERMES_FUNDAMENTALS_ROLE,
    DEEPHERMES_REASONING_ROLE,
    FINANCE_QLORA_ROLE,
    NAMED_ROLES,
    NamedSpecialist,
    SpecialistOutput,
    build_named_specialists,
    build_provider_map_from_orchestrator,
    run_named_specialists,
)
__all__ = [
    # base
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "FeatherlessProvider",
    # adapter
    "AgentLLMAdapter",
    "extract_json_object",
    # orchestrator
    "FeatherlessOrchestrator",
    "FailureKind",
    "ProviderSpec",
    "UsageLog",
    "UsageRecord",
    "classify_failure",
    "load_provider_specs",
    # snapshot
    "build_snapshot",
    "pre_screen",
    "PreScreenResult",
    # named specialists
    "DEEPHERMES_REASONING_ROLE",
    "DEEPHERMES_FUNDAMENTALS_ROLE",
    "FINANCE_QLORA_ROLE",
    "NAMED_ROLES",
    "NamedSpecialist",
    "SpecialistOutput",
    "build_named_specialists",
    "build_provider_map_from_orchestrator",
    "run_named_specialists",
]
