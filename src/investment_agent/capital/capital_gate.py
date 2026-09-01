"""Seven-State Capital Gate — Quantitative Intelligence Layer for X Quant X.

WHAT
====
Evaluates the 7-dimensional investment state vector (x_1..x_7), portfolio allocations, risk thresholds,
and regime constraints to determine capital deployment permission (pass/fail), gated leverage, and allocation multiplier.

WHY
===
Ensures total capital preservation by gating position allocation through strict analytical thresholds, soft rule penalty
reductions, and hard circuit-breaker constraints before orders reach execution systems.

HOW
===
- Seven State Vector: x = [s_portfolio, s_capital, s_reputation, s_state, s_regime, s_uncertainty, s_disagreement]ᵀ ∈ ℝ⁷.
- State-of-Charge & Capital Capacity: Bounds portfolio risk capacity relative to historical drawdown and volatility bounds.
- Soft & Hard Rule Gating: Soft rules (e.g. regime disagreement, drawdowns) apply multiplicative allocation reductions;
  hard rules (e.g. max drawdown, invalid states) trigger immediate circuit breaker rejection (gate_passed = False).
- Pure Analytical Contract: Pure evaluation function returns CapitalGateResult; performs zero trading or API side-effects.

Mathematical Specifications
===========================
Specified in:
    - high_level_proofs/finite_investment_architecture_states_of_portfolio_investments_securities_finance_markets_fundamentals_sectors.md
    - high_level_proofs/high_level_kalman_filter_states_capital_allocation_proof.tex
    - alpaca_paper_trading_specifications_x_quant_x/028_xquantx_risk_ruleset.txt
    - alpaca_paper_trading_specifications_x_quant_x/017_xquantx_scoring_rules.txt

Architectural Role
==================
Analytical gating engine. Consumes KalmanState (kalman_filter.py), AgentReputationTracker (agent_reputation.py),
and EnsembleAggregate (ensemble_signal.py) outputs to gate risk before execution.
Performs no order placement, broker API calls, or external side-effects.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Mapping

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from pathlib import Path

from ..filters.kalman_filter import KalmanState
from ..signals.ensemble_signal import (
    AgentOutput,
    EnsembleAggregate,
    compute_ensemble_aggregate,
)
from ..filters.investment_kalman_gain import compute_investment_kalman_gain
from ..regimes.regimes import VALID_REGIMES


_DEFAULT_STATE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "economic":    {"minimum": 0.15, "full": 0.70},
    "financial":   {"minimum": 0.20, "full": 0.80},
    "fiscal":      {"minimum": 0.10, "full": 0.65},
    "portfolio":   {"minimum": 0.20, "full": 0.70},
    "fundamental": {"minimum": 0.15, "full": 0.75},
    "market":      {"minimum": 0.20, "full": 0.80},
    "sector":      {"minimum": 0.15, "full": 0.75},
}


# Drawdown circuit-breaker thresholds. These are the canonical risk
# rule bounds and are exposed at module scope so other components
# (e.g. the dashboard) can read them authoritatively.
DRAWDOWN_FLATTEN_PCT: float = 0.15
DRAWDOWN_REDUCE_PCT: float = 0.10


def _load_risk_thresholds_from_path(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """Load state thresholds from config/risk_rules.toml if present; otherwise fall back to canonical defaults."""
    if path is None or not path.exists() or not path.is_file():
        return {
            key: {"minimum": float(val["minimum"]), "full": float(val["full"])}
            for key, val in _DEFAULT_STATE_THRESHOLDS.items()
        }

    try:
        with path.open("rb") as fp:
            raw = tomllib.load(fp)
    except Exception:
        return {
            key: {"minimum": float(val["minimum"]), "full": float(val["full"])}
            for key, val in _DEFAULT_STATE_THRESHOLDS.items()
        }

    state_cfg = raw.get("state_thresholds") if (raw.get("state_thresholds") and isinstance(raw.get("state_thresholds"), dict)) else raw
    if not isinstance(state_cfg, dict):
        return {
            key: {"minimum": float(val["minimum"]), "full": float(val["full"])}
            for key, val in _DEFAULT_STATE_THRESHOLDS.items()
        }

    result: Dict[str, Dict[str, float]] = {}
    for key, fallback in _DEFAULT_STATE_THRESHOLDS.items():
        entry = state_cfg.get(key, {})
        if not isinstance(entry, dict):
            result[key] = {"minimum": float(fallback["minimum"]), "full": float(fallback["full"])}
            continue
        minimum = entry.get("minimum", fallback["minimum"])
        full = entry.get("full", fallback["full"])
        result[key] = {
            "minimum": float(minimum),
            "full": float(full),
        }
    return result


def _resolve_risk_thresholds_path() -> Optional[Path]:
    """Look for a repo-local config file without creating additional project files."""
    candidates = [
        Path(__file__).resolve().parent / "config" / "risk_rules.toml",
        Path.cwd() / "config" / "risk_rules.toml",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


STATE_THRESHOLDS = _load_risk_thresholds_from_path(_resolve_risk_thresholds_path())


# ---------------------------------------------------------------------------
# Risk Verdict Enum
# ---------------------------------------------------------------------------

class RiskVerdict(Enum):
    """Authoritative Risk-Gate Verdict options."""
    ALLOW = "ALLOW"
    REDUCE = "REDUCE"
    BLOCK = "BLOCK"
    FLATTEN = "FLATTEN"


# ---------------------------------------------------------------------------
# Seven State Vector Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SevenStateVector:
    """Immutable representation of the 7-dimensional investment State-of-Charge.

    All state charge dimensions S^(d) represent normalized energy / capacity levels
    in [0.0, 1.0].

    Attributes
    ----------
    economic : float
        Economic state charge E_t ∈ [0, 1] (GDP engine / macro growth).
    financial : float
        Financial state charge F_t ∈ [0, 1] (Market plumbing / stress resilience).
    fiscal : float
        Fiscal state charge G_t ∈ [0, 1] (Stimulus reservoir / policy capacity).
    portfolio : float
        Portfolio state charge P_t ∈ [0, 1] (Capital resilience / drawdown headroom).
    fundamental : float
        Fundamental state charge U_t ∈ [0, 1] (Valuation health / earnings sanity).
    market : float
        Market state charge M_t ∈ [0, 1] (Microstructure quality / liquidity).
    sector : float
        Sector / Technology state charge T_t ∈ [0, 1] (S-curve adoption momentum).
    """
    economic: float
    financial: float
    fiscal: float
    portfolio: float
    fundamental: float
    market: float
    sector: float

    def __post_init__(self) -> None:
        """Strict validation of all 7 state dimensions upon initialization."""
        field_map = {
            "economic": self.economic,
            "financial": self.financial,
            "fiscal": self.fiscal,
            "portfolio": self.portfolio,
            "fundamental": self.fundamental,
            "market": self.market,
            "sector": self.sector,
        }

        for name, val in field_map.items():
            if isinstance(val, bool):
                raise TypeError(f"State '{name}' cannot be a boolean value: {val}")
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"State '{name}' must be numeric (float or int), got {type(val).__name__}"
                )
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"State '{name}' cannot be NaN or Infinity: {val}")
            if val < 0.0 or val > 1.0:
                raise ValueError(
                    f"State '{name}' must be inside [0.0, 1.0], got {val}"
                )
            # G-012: Normalize integer values to float for consistency with AgentOutput
            val_float = float(val)
            object.__setattr__(self, name, val_float)

    @classmethod
    def full_charge(cls) -> "SevenStateVector":
        """Create a fully charged state explicitly instead of relying on implicit defaults."""
        return cls(
            economic=1.0,
            financial=1.0,
            fiscal=1.0,
            portfolio=1.0,
            fundamental=1.0,
            market=1.0,
            sector=1.0,
        )


# ---------------------------------------------------------------------------
# Capital Gate Result Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapitalGateResult:
    """Immutable result emitted by SevenStateCapitalGate.evaluate().
    
    G-013: All mutable collection fields are wrapped as read-only views.
    - state_charges and state_gatings are MappingProxyType (immutable dict views)
    - triggered_rules is a tuple (immutable sequence)
    - The dataclass itself is frozen, preventing field reassignment

    Attributes
    ----------
    verdict : RiskVerdict
        The overall risk-gate verdict (ALLOW, REDUCE, BLOCK, FLATTEN).
    gating_factor : float
        Composite product-of-gating-functions G(S_t) ∈ [0.0, 1.0].
    effective_cap : float
        Combined deployment cap K_t * G(S_t) * reduce_factor ∈ [0.0, 1.0].
        On BLOCK/FLATTEN this is forced to 0.0.
    reduce_factor : float
        Multiplicative penalty from REDUCE rules ∈ [0.0, 1.0]. 1.0 means no reduction.
    state_charges : Mapping[str, float]
        Immutable view of input state charge values S^(d).
    state_gatings : Mapping[str, float]
        Immutable view of individual piecewise-linear gating values g_d(S^(d)).
    triggered_rules : tuple
        Immutable sequence of rule identifiers triggered during evaluation (e.g. CONC-001, DD-001).
    reason : str
        Human-readable summary of evaluation rationale and rule triggers.
    kalman_gain : float
        Investment Kalman gain K_t computed from ensemble effective confidence and disagreement.
    ensemble_agg : EnsembleAggregate
        Immutable ensemble aggregate used for this evaluation (chain-of-custody).
    """
    verdict: RiskVerdict
    gating_factor: float
    effective_cap: float
    reduce_factor: float
    state_charges: Mapping[str, float]
    state_gatings: Mapping[str, float]
    triggered_rules: tuple
    reason: str
    kalman_gain: float
    ensemble_agg: EnsembleAggregate


# ---------------------------------------------------------------------------
# Authoritative State Thresholds
# ---------------------------------------------------------------------------

STATE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "economic":    {"minimum": 0.15, "full": 0.70},
    "financial":   {"minimum": 0.20, "full": 0.80},
    "fiscal":      {"minimum": 0.10, "full": 0.65},
    "portfolio":   {"minimum": 0.20, "full": 0.70},
    "fundamental": {"minimum": 0.15, "full": 0.75},
    "market":      {"minimum": 0.20, "full": 0.80},
    "sector":      {"minimum": 0.15, "full": 0.75},
}


# ---------------------------------------------------------------------------
# Core Seven-State Capital Gate Engine
# ---------------------------------------------------------------------------

def compute_individual_gating(state_name: str, value: float) -> float:
    """Compute the piecewise-linear gating function g_d(S) for a state dimension.
    
    Boundary convention: S == minimum returns 0.0 via the linear interpolation
    branch ((value - minimum) / (full - minimum) = 0.0). This matches the
    inclusive lower-bound interpretation (S <= minimum → gate = 0) documented
    in the authoritative specification.
    """
    if not isinstance(state_name, str):
        raise TypeError(f"state_name must be a string, got {type(state_name).__name__}")
    if state_name not in STATE_THRESHOLDS:
        raise ValueError(
            f"Unknown state dimension '{state_name}'. "
            f"Must be one of {sorted(STATE_THRESHOLDS.keys())}"
        )

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"State value for '{state_name}' must be numeric, got {type(value).__name__}")

    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"State value for '{state_name}' cannot be NaN or Infinity: {value}")

    thresh = STATE_THRESHOLDS[state_name]
    minimum = thresh["minimum"]
    full = thresh["full"]

    if value >= full:
        return 1.0
    elif value < minimum:
        return 0.0
    else:
        return (value - minimum) / (full - minimum)


def compute_gating_factor(states: SevenStateVector) -> Tuple[float, Dict[str, float]]:
    """Compute composite gating factor G(S_t) as product of all 7 individual gates.

    Parameters
    ----------
    states : SevenStateVector
        The 7-dimensional state charge vector.

    Returns
    -------
    gating_factor : float
        Product of all seven individual gating factors, bounded in [0.0, 1.0].
    state_gatings : Dict[str, float]
        Dictionary mapping each state dimension name to its individual gating value.
    """
    state_dict = {
        "economic": states.economic,
        "financial": states.financial,
        "fiscal": states.fiscal,
        "portfolio": states.portfolio,
        "fundamental": states.fundamental,
        "market": states.market,
        "sector": states.sector,
    }

    gatings: Dict[str, float] = {}
    product = 1.0

    for name, val in state_dict.items():
        g_val = compute_individual_gating(name, val)
        gatings[name] = g_val
        product *= g_val

    # Guarantee result stays strictly within [0.0, 1.0]
    composite_gating = max(0.0, min(1.0, float(product)))
    return composite_gating, gatings


# ---------------------------------------------------------------------------
# Helper Validators for Portfolio Context
# ---------------------------------------------------------------------------

KNOWN_PORTFOLIO_CONTEXT_KEYS = {
    "position_pct",
    "gross_leverage",
    "entropy",
    "drawdown_pct",
    "execution_timeout_seconds",
    "sector_exposure_pct",
    "is_new_long",
    "regime",
    "session_peak_equity",
    "current_equity",
    "available_liquidity",
}

# VALID_REGIMES is imported from regimes.py (line 46) — do NOT redefine here.


def _parse_bool(val: Any, field_name: str, default: bool = False) -> bool:
    """Safely parse boolean values without Python truthiness vulnerabilities."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val == 1 or val == 1.0:
            return True
        elif val == 0 or val == 0.0:
            return False
        else:
            raise ValueError(f"Numeric boolean field '{field_name}' must be 0 or 1, got {val}")
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ("true", "1", "yes", "t"):
            return True
        elif cleaned in ("false", "0", "no", "f"):
            return False
        else:
            raise ValueError(f"String boolean field '{field_name}' has invalid representation: '{val}'")
    raise TypeError(f"Field '{field_name}' must be boolean, numeric 0/1, or boolean string, got {type(val).__name__}")


def _parse_float(
    val: Any,
    field_name: str,
    default: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    required: bool = False
) -> Optional[float]:
    """Strictly parse and validate float values from portfolio context.
    
    Parameters
    ----------
    val : Any
        The value to parse.
    field_name : str
        Name of the field (for error messages).
    default : Optional[float]
        Default value if val is None. If required=True and val is None and default is None, raises error.
    min_val : Optional[float]
        Minimum acceptable value (inclusive).
    max_val : Optional[float]
        Maximum acceptable value (inclusive).
    required : bool
        If True, val cannot be None and no default is used. Raises error on missing data (G-004).
        
    Raises
    ------
    ValueError
        If required=True and val is None (preventing fail-open defaults).
    """
    if val is None:
        if required:
            raise ValueError(
                f"Field '{field_name}' is required for risk gating. "
                f"Missing critical risk context is not acceptable (G-004: fail-stop behavior)."
            )
        res = default
    elif isinstance(val, bool):
        raise TypeError(f"Field '{field_name}' cannot be a boolean: {val}")
    elif isinstance(val, (int, float)):
        res = float(val)
    elif isinstance(val, str):
        try:
            res = float(val)
        except ValueError:
            raise ValueError(f"Field '{field_name}' string could not be converted to float: '{val}'")
    else:
        raise TypeError(f"Field '{field_name}' must be numeric or numeric string, got {type(val).__name__}")

    if math.isnan(res) or math.isinf(res):
        raise ValueError(f"Field '{field_name}' cannot be NaN or Infinity: {res}")

    if min_val is not None and res < min_val:
        raise ValueError(f"Field '{field_name}' cannot be less than {min_val}, got {res}")

    if max_val is not None and res > max_val:
        raise ValueError(f"Field '{field_name}' cannot be greater than {max_val}, got {res}")

    return res


def evaluate(
    kalman_state: KalmanState,
    states: SevenStateVector,
    portfolio_context: Dict[str, Any],
    agents: List[AgentOutput],
    agent_weights: Dict[str, float],
    sigma_base_squared: float = 1.0,
    ensemble_agg: Optional[EnsembleAggregate] = None,
) -> CapitalGateResult:
    r"""Evaluate the Seven-State Capital Gate and return an authoritative RiskVerdict.

    This function is pure: it does not mutate any inputs or external system state.

    Parameters
    ----------
    kalman_state : KalmanState
        Posterior state estimate snapshot from KalmanFilter.
    states : SevenStateVector
        7-dimensional investment State-of-Charge vector.
    portfolio_context : Dict[str, Any]
        Portfolio risk context data (drawdown, concentration, leverage, regime, etc.).
    agents : List[AgentOutput]
        List of agent quantitative outputs for the current period.
    agent_weights : Dict[str, float]
        Dictionary mapping agent_id to positive reputation weight w_i.
    sigma_base_squared : float, optional
        Base model variance σ²_base > 0.0 (default 1.0).
    ensemble_agg : Optional[EnsembleAggregate], optional
        Pre-computed ensemble aggregate. If provided, the gate uses this object
        directly instead of recomputing from agents/weights. This preserves
        chain-of-custody for downstream audit.

    Returns
    -------
    CapitalGateResult
        Immutable result containing verdict, gating factor, effective cap, state breakdown,
        and rule triggers.
        
    Raises
    ------
    ValueError
        If sigma_base_squared <= 0 (G-003: validation requirement).
    """
    # -----------------------------------------------------------------------
    # 0. Validate sigma_base_squared (G-003: Critical validation)
    # -----------------------------------------------------------------------
    if isinstance(sigma_base_squared, bool) or not isinstance(sigma_base_squared, (int, float)):
        raise TypeError(f"sigma_base_squared must be numeric, got {type(sigma_base_squared).__name__}")
    
    sigma_sq_float = float(sigma_base_squared)
    if math.isnan(sigma_sq_float) or math.isinf(sigma_sq_float):
        raise ValueError(f"sigma_base_squared cannot be NaN or Infinity: {sigma_sq_float}")
    
    if sigma_sq_float <= 0.0:
        raise ValueError(
            f"sigma_base_squared must be strictly positive (> 0) per contract, "
            f"got {sigma_sq_float}"
        )
    
    # -----------------------------------------------------------------------
    # 1. Validate types and basic input structure
    # -----------------------------------------------------------------------
    if not isinstance(agents, list):
        raise TypeError(f"agents must be a list, got {type(agents).__name__}")
    if len(agents) != 7:
        raise ValueError(
            f"Seven-state capital gate requires exactly 7 agents, got {len(agents)}. "
            f"This architecture is defined around the seven-dimensional state vector."
        )

    if not isinstance(agent_weights, dict):
        raise TypeError(f"agent_weights must be a dict, got {type(agent_weights).__name__}")

    if not isinstance(kalman_state, KalmanState):
        raise TypeError(f"kalman_state must be a KalmanState, got {type(kalman_state).__name__}")

    # Accept either a SevenStateVector instance or any duck-typed object with the
    # seven required numeric attributes. Using duck-typing avoids false negatives
    # when test code constructs compatible objects.
    required_state_attrs = [
        "economic",
        "financial",
        "fiscal",
        "portfolio",
        "fundamental",
        "market",
        "sector",
    ]
    for attr in required_state_attrs:
        if not hasattr(states, attr):
            raise TypeError(f"states must provide attribute '{attr}' for SevenStateVector compatibility")
        val = getattr(states, attr)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"State '{attr}' must be numeric (float or int), got {type(val).__name__}")
        # Validate finite and in-range [0.0, 1.0] per authoritative SevenStateVector contract
        try:
            val_float = float(val)
        except Exception:
            raise TypeError(f"State '{attr}' must be numeric (float or int), got {type(val).__name__}")

        if math.isnan(val_float) or math.isinf(val_float):
            raise ValueError(f"State '{attr}' cannot be NaN or Infinity: {val}")

        if val_float < 0.0 or val_float > 1.0:
            raise ValueError(f"State '{attr}' must be inside [0.0, 1.0], got {val}")

    if not isinstance(portfolio_context, dict):
        raise TypeError(f"portfolio_context must be a dict, got {type(portfolio_context).__name__}")

    # -----------------------------------------------------------------------
    # 2. Compute Ensemble Aggregate (G-005: Atomic aggregation prevents inconsistency)
    # -----------------------------------------------------------------------
    # Use pre-computed ensemble if provided (preserves chain-of-custody);
    # otherwise compute atomically from agents/weights.
    if ensemble_agg is not None:
        if not isinstance(ensemble_agg, EnsembleAggregate):
            raise TypeError(f"ensemble_agg must be an EnsembleAggregate, got {type(ensemble_agg).__name__}")
        _ensemble_signal = ensemble_agg.ensemble_signal
        _disagreement = ensemble_agg.disagreement
        _effective_confidence = ensemble_agg.effective_confidence
    else:
        ensemble_agg = compute_ensemble_aggregate(agents, agent_weights)
        _ensemble_signal = ensemble_agg.ensemble_signal
        _disagreement = ensemble_agg.disagreement
        _effective_confidence = ensemble_agg.effective_confidence

    # NOTE: kalman_state.price_variance is used as a single-asset variance proxy for P_{t|t-1}
    k_t = compute_investment_kalman_gain(
        prediction_covariance=kalman_state.price_variance,
        effective_confidence=_effective_confidence,
        disagreement=_disagreement,
        sigma_base_squared=sigma_sq_float,
    )

    # -----------------------------------------------------------------------
    # 3. Extract Portfolio Context with Strict Validation (G-004: Prevent fail-open)
    # -----------------------------------------------------------------------
    unknown_keys = set(portfolio_context.keys()) - KNOWN_PORTFOLIO_CONTEXT_KEYS
    if unknown_keys:
        raise KeyError(
            f"Unknown portfolio_context key(s): {sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(KNOWN_PORTFOLIO_CONTEXT_KEYS)}"
        )

    # G-004: Require critical risk fields to prevent fail-open defaults.
    # These fields are essential for risk gate decision-making and must be explicitly provided.
    position_pct = _parse_float(
        portfolio_context.get("position_pct"), "position_pct",
        default=None, min_val=0.0, max_val=1.0, required=True
    )
    gross_leverage = _parse_float(
        portfolio_context.get("gross_leverage"), "gross_leverage",
        default=None, min_val=0.0, required=True
    )
    entropy = _parse_float(
        portfolio_context.get("entropy"), "entropy",
        default=None, min_val=0.0, max_val=1.0, required=True
    )
    drawdown_pct = _parse_float(
        portfolio_context.get("drawdown_pct"), "drawdown_pct",
        default=None, min_val=0.0, max_val=1.0, required=True
    )
    execution_timeout = _parse_float(
        portfolio_context.get("execution_timeout_seconds"), "execution_timeout_seconds",
        default=None, min_val=0.0, required=True
    )
    sector_exposure_pct = _parse_float(
        portfolio_context.get("sector_exposure_pct"), "sector_exposure_pct",
        default=None, min_val=0.0, max_val=1.0, required=True
    )

    available_liquidity = _parse_float(
        portfolio_context.get("available_liquidity"), "available_liquidity",
        default=None, min_val=0.0, required=True
    )

    is_new_long = _parse_bool(portfolio_context.get("is_new_long"), "is_new_long", False)

    regime_raw = portfolio_context.get("regime", "R01")
    if regime_raw is None:
        regime = "R01"
    elif isinstance(regime_raw, str):
        regime = regime_raw.strip().upper()
    else:
        raise TypeError(f"Field 'regime' must be a string identifier, got {type(regime_raw).__name__}")

    if regime not in VALID_REGIMES:
        raise ValueError(f"Invalid regime identifier: '{regime_raw}'. Must be one of {sorted(VALID_REGIMES)}")

    # Optional fields: session peak/current equity for drawdown calculation
    session_peak_raw = portfolio_context.get("session_peak_equity")
    current_equity_raw = portfolio_context.get("current_equity")

    session_peak = _parse_float(session_peak_raw, "session_peak_equity", None, min_val=0.0) if session_peak_raw is not None else None
    current_equity = _parse_float(current_equity_raw, "current_equity", None, min_val=0.0) if current_equity_raw is not None else None

    if session_peak is not None and current_equity is not None:
        if session_peak == 0.0:
            raise ValueError("session_peak_equity cannot be zero when current_equity is provided")
        if current_equity > session_peak:
            raise ValueError(
                f"current_equity ({current_equity}) cannot exceed session_peak_equity ({session_peak}): "
                f"session peak must be >= current equity by definition"
            )
        computed_dd = (session_peak - current_equity) / session_peak
        drawdown_pct = max(drawdown_pct, computed_dd)

    # -----------------------------------------------------------------------
    # 4. Compute Composite Gating & Effective Cap
    # -----------------------------------------------------------------------
    gating_factor, state_gatings = compute_gating_factor(states)
    raw_cap = max(0.0, min(1.0, float(k_t * gating_factor)))
    effective_cap = raw_cap

    # Record state charges dictionary
    state_charges = {
        "economic": states.economic,
        "financial": states.financial,
        "fiscal": states.fiscal,
        "portfolio": states.portfolio,
        "fundamental": states.fundamental,
        "market": states.market,
        "sector": states.sector,
    }

    # -----------------------------------------------------------------------
    # 5. Risk Verdict Hierarchy Evaluation
    # -----------------------------------------------------------------------
    flatten_rules: List[str] = []
    block_rules: List[str] = []
    reduce_rules: List[str] = []

    # --- A. FLATTEN Triggers (Highest Priority) ---
    # Rule DD-001: Drawdown > DRAWDOWN_FLATTEN_PCT (canonical 15%)
    if drawdown_pct > DRAWDOWN_FLATTEN_PCT:
        flatten_rules.append("DD-001")

    # Rule EXEC-001: Execution timeout > 30 seconds
    if execution_timeout > 30.0:
        flatten_rules.append("EXEC-001")

    # --- B. BLOCK Triggers ---
    # State charge below minimum threshold
    for name, g_val in state_gatings.items():
        if g_val == 0.0:
            block_rules.append(f"GATE-{name.upper()}-MIN")

    # Rule LIQ-001: Liquidity floor < $5,000
    if available_liquidity < 5000.0:
        block_rules.append("LIQ-001")

    # Rule CONC-001: Concentration cap > 20% (0.20)
    if position_pct > 0.20:
        block_rules.append("CONC-001")

    # Rule LEV-001: Gross leverage cap > 1.0
    if gross_leverage > 1.0:
        block_rules.append("LEV-001")

    # Rule REGM-001: Bear capitulation (R04) or Macro shock (R07) new long without > 0.85 effective confidence
    if is_new_long and regime in ("R04", "R07") and _effective_confidence <= 0.85:
        block_rules.append("REGM-001")

    # Rule ENT-002: Hard entropy block > 0.90
    if entropy > 0.90:
        block_rules.append("ENT-002")

    # --- C. REDUCE Triggers & Quantitative Scaling ---
    # Multiplicative reduce_factor accumulator (1.0 = no reduction)
    reduce_factor = 1.0

    # Any state charge in caution range (minimum <= S < full)
    # NOTE: These are already encoded in G(S_t) via the gating product.
    for name, g_val in state_gatings.items():
        if 0.0 < g_val < 1.0:
            reduce_rules.append(f"GATE-{name.upper()}-CAUTION")

    # Rule SECT-001: Sector exposure > 35% (0.35)
    # Whitepaper §1.7.1: "REDUCE the proposed position to bring sector total back to <= 0.35"
    if sector_exposure_pct > 0.35:
        reduce_rules.append("SECT-001")
        reduce_factor *= 0.35 / sector_exposure_pct

    # Rule ENT-001: Soft entropy reduction > 0.75 (and <= 0.90)
    # Whitepaper §1.7.3: "scale all proposed new positions by (1 - U_t)"
    if 0.75 < entropy <= 0.90:
        reduce_rules.append("ENT-001")
        reduce_factor *= (1.0 - entropy)

    # Rule ECONF-001: Low ensemble effective confidence < 0.40
    # Whitepaper §1.7.2: "scale the proposed position by (confidence / 0.40)"
    if _effective_confidence < 0.40:
        reduce_rules.append("ECONF-001")
        reduce_factor *= _effective_confidence / 0.40

    # Rule DISAG-001: High ensemble disagreement > 0.50
    # Quantitative penalty: scale by (1 - D_t), floored at 0.
    if _disagreement > 0.50:
        reduce_rules.append("DISAG-001")
        reduce_factor *= max(0.0, 1.0 - _disagreement)

    reduce_factor = max(0.0, min(1.0, reduce_factor))

    # -----------------------------------------------------------------------
    # 6. Determine Verdict, Reason, Apply Reduce Factor, Enforce Zero Cap
    # -----------------------------------------------------------------------
    if flatten_rules:
        verdict = RiskVerdict.FLATTEN
        triggered = flatten_rules
        reason = f"FLATTEN triggered by portfolio risk breach: {', '.join(flatten_rules)}"
        effective_cap = 0.0
    elif block_rules:
        verdict = RiskVerdict.BLOCK
        triggered = block_rules
        reason = f"BLOCK triggered by hard risk rule limit: {', '.join(block_rules)}"
        effective_cap = 0.0
    elif reduce_rules:
        verdict = RiskVerdict.REDUCE
        triggered = reduce_rules
        effective_cap = max(0.0, min(1.0, raw_cap * reduce_factor))
        reason = (
            f"REDUCE triggered by soft limit / state caution: {', '.join(reduce_rules)}. "
            f"reduce_factor={reduce_factor:.6f}, effective_cap={effective_cap:.6f}"
        )
    else:
        verdict = RiskVerdict.ALLOW
        triggered = []
        reason = "ALLOW: All 7 state charges at full capacity, no risk rule violations"
        effective_cap = raw_cap

    return CapitalGateResult(
        verdict=verdict,
        gating_factor=gating_factor,
        effective_cap=effective_cap,
        reduce_factor=reduce_factor,
        state_charges=types.MappingProxyType(state_charges),
        state_gatings=types.MappingProxyType(state_gatings),
        triggered_rules=tuple(triggered),
        reason=reason,
        kalman_gain=k_t,
        ensemble_agg=ensemble_agg,
    )

