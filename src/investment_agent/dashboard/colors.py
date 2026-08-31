"""X Quant X dashboard color palette tokens.

WHAT
====
Central color token definitions matching
alpaca_paper_trading_specifications_x_quant_x/022_xquantx_visualization_dev_standards.txt
Section 1.7. No other module should hardcode a hex color -- import from here.

WHY
===
The spec reserves ALERT_CRITICAL exclusively for ALERT states (REDUCE/BLOCK/FLATTEN
verdicts, drawdown > 15%, stress regimes). Keeping the tokens in one place makes that
rule easy to audit and impossible to accidentally violate with a copy-pasted hex code.
"""

BACKGROUND_PRIMARY = "#1a1a2e"
TEXT_PRIMARY = "#e0e0e0"
GRID_LINE = "#2a2a4a"
SERIES_EQUITY = "#7ec8e3"
SERIES_BENCHMARK = "#4a4a6a"
ALERT_WARN = "#FF9800"
ALERT_CRITICAL = "#FFACE9"  # reserved: REDUCE/BLOCK/FLATTEN, drawdown >15%, R04/R07 only
REGIME_BULL = "#4CAF50"
REGIME_BEAR = "#F44336"
REGIME_NEUTRAL = "#9E9E9E"

# Per-regime color mapping (022 doc Section 1.6.3)
REGIME_COLORS = {
    "R01": REGIME_BULL, "R02": REGIME_BULL, "R11": REGIME_BULL,
    "R03": ALERT_WARN, "R04": ALERT_CRITICAL,
    "R05": REGIME_BEAR, "R07": ALERT_CRITICAL,
    "R06": REGIME_NEUTRAL, "R08": REGIME_NEUTRAL, "R09": REGIME_NEUTRAL,
    "R10": REGIME_NEUTRAL, "R12": REGIME_NEUTRAL,
}

# Verdicts that must render with the alert-critical token (spec: REDUCE/BLOCK/FLATTEN)
ALERT_VERDICTS = {"BLOCK", "FLATTEN", "REDUCE"}

VERDICT_ROW_COLOR = {
    "BLOCK": ALERT_WARN,
    "FLATTEN": ALERT_CRITICAL,
    "REDUCE": "#FFB74D",  # light orange, per 004 doc P06 rule
    "ALLOW": BACKGROUND_PRIMARY,
}

ORDER_STATUS_COLOR = {
    "filled": REGIME_BULL,
    "cancelled": REGIME_NEUTRAL,
    "new": SERIES_EQUITY,
    "partially_filled": ALERT_WARN,
}

RULESET_VERSION = "xquantx-2026.08"

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BACKGROUND_PRIMARY,
    plot_bgcolor=BACKGROUND_PRIMARY,
    font=dict(color=TEXT_PRIMARY),
    xaxis=dict(gridcolor=GRID_LINE, zerolinecolor=GRID_LINE),
    yaxis=dict(gridcolor=GRID_LINE, zerolinecolor=GRID_LINE),
    margin=dict(l=50, r=30, t=50, b=40),
)
