"""Persistence helpers for ``AgentReputationTracker``.

The tracker is in-memory by design. To survive a process restart (and
so the dashboard can read real ``alpha``/``beta`` instead of the
hard-coded 1.0/1.0 fallback), we round-trip the tracker's
``to_dict`` / ``from_dict`` to a single JSON file. The save is atomic
(tmp+rename) so a crash mid-write never leaves a corrupt file.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from .agent_reputation import AgentReputationTracker


DEFAULT_REPUTATION_FILE = "reputation_state.json"


def save_reputation(
    tracker: AgentReputationTracker,
    path: str | os.PathLike = DEFAULT_REPUTATION_FILE,
) -> None:
    """Atomically write the tracker's serialized state to ``path``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = tracker.to_dict()
    fd, tmp_path = tempfile.mkstemp(
        prefix=p.name + ".", suffix=".tmp", dir=str(p.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, p)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_reputation(
    path: str | os.PathLike = DEFAULT_REPUTATION_FILE,
) -> Optional[AgentReputationTracker]:
    """Load a previously saved tracker. Returns ``None`` if no file exists
    or the file is corrupt (so the caller can fall back to a fresh
    in-memory tracker)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return AgentReputationTracker.from_dict(data)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_REPUTATION_FILE",
    "load_reputation",
    "save_reputation",
]
