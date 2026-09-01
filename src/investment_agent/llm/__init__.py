"""LLM package — providers and adapters for the X Quant X agent layer.

Public API:
    LLMProvider      — abstract protocol
    LLMResponse      — typed result of a single LLM call
    MockLLMProvider  — deterministic in-process provider (offline / tests)
    FeatherlessProvider — Featherless chat-completions HTTP backend
    AgentLLMAdapter  — bridges LLM output → ``AgentOutput``
    extract_json_object — defensive JSON extractor
"""

from .base import (
    LLMProvider,
    LLMResponse,
    MockLLMProvider,
    FeatherlessProvider,
)
from .adapter import AgentLLMAdapter, extract_json_object

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "FeatherlessProvider",
    "AgentLLMAdapter",
    "extract_json_object",
]
