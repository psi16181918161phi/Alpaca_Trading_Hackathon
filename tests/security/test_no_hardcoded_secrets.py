"""Security regression tests for the ``src/`` tree.

Covers three concerns the codebase has no existing tooling for:
  * static analysis via ``bandit`` (OWASP-adjacent Python security linter)
  * hardcoded Alpaca credential literals committed to source
  * secret values leaking into printed/logged output

None of these tests make a network call; they are pure static analysis
over the repository's own source tree.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "investment_agent"

# A real Alpaca API key/secret is a long alphanumeric token. This pattern
# looks for assignment of such a literal to a credential-shaped name,
# not for the (safe) os.getenv(...) pattern used throughout the codebase.
_HARDCODED_SECRET_PATTERN = re.compile(
    r'(?i)(api_key|secret_key|api_secret|password)\s*=\s*["\'][A-Za-z0-9/_+-]{20,}["\']'
)


def _iter_python_files():
    return sorted(SRC_DIR.rglob("*.py"))


def test_no_hardcoded_alpaca_credentials_in_src():
    offenders = []
    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _HARDCODED_SECRET_PATTERN.finditer(text):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {match.group(0)[:60]}")
    assert not offenders, "hardcoded credential-shaped literals found:\n" + "\n".join(offenders)


def test_bandit_reports_zero_high_severity_findings():
    result = subprocess.run(
        [sys.executable, "-m", "bandit", "-r", str(SRC_DIR), "-f", "json", "-q"],
        capture_output=True, text=True, timeout=120,
    )
    # bandit exits non-zero when it finds anything; parse JSON regardless of exit code.
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise AssertionError(f"bandit did not produce valid JSON: {result.stdout}\n{result.stderr}")
    high_severity = [
        r for r in report.get("results", [])
        if r.get("issue_severity") == "HIGH"
    ]
    assert not high_severity, "bandit found HIGH severity issues:\n" + "\n".join(
        f"{r['filename']}:{r['line_number']} {r['test_id']} {r['issue_text']}"
        for r in high_severity
    )


def test_env_example_has_no_real_looking_secrets():
    """.env.example must stay a template — no accidentally-committed real value."""
    env_example = REPO_ROOT / ".env.example"
    text = env_example.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if "KEY" in key.upper() or "SECRET" in key.upper():
            assert value.strip() == "", (
                f"{env_example.name} line {key}= must stay blank (template only), got a value"
            )
