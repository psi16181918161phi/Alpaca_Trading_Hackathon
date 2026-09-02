# Mutation Testing Scope

Mutation testing (via `mutmut`) is configured in the repository-root
`pyproject.toml` under `[tool.mutmut]`.

## Scope

The first mutation-testing pass is intentionally bounded to two modules,
per `CHATS/2026-09-02_archive-consolidation-regression-suite-PLAN.md`
Task P2-04:

- `src/investment_agent/capital/capital_gate.py` -- the seven-state
  capital risk-verdict gate.
- `src/investment_agent/products/product_gate.py` -- the equity/option/
  crypto/none vehicle selector.

These are the highest-value, smallest-surface risk-verdict modules in the
codebase, chosen so a single `mutmut run` stays fast and its
surviving-mutant list stays reviewable, rather than mutating the entire
`src/investment_agent/` tree in one unbounded pass.

## Running

```powershell
python -m mutmut run
python -m mutmut results
python -m mutmut html   # writes html/ with per-mutant detail
```

## Expanding scope

To mutate additional modules in a future pass, add their path to
`paths_to_mutate` in `pyproject.toml` and extend `runner` to include the
corresponding test directories, so surviving mutants are attributable to
a real test gap rather than an out-of-scope test suite.
