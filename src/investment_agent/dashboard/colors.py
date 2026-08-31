"""X Quant X dashboard color, typography, and layout tokens.

WHAT
====
Central token definitions. No other module should hardcode a hex color or
font stack -- import from here.

WHY
===
ALERT_CRITICAL is reserved exclusively for ALERT states (REDUCE/BLOCK/
FLATTEN verdicts, drawdown > 15%, stress regimes). Keeping the tokens in one
place makes that rule easy to audit and impossible to accidentally violate
with a copy-pasted hex code.

BRAND COLOR SOURCE
===================
This palette matches the "Variant A / Variant B" system psi described in
Discord and the mockup generated from it: a near-black base with a dusty
rose/mauve (#B76E79) accent for normal-state charts and UI (Variant A), and
a bold pink (#FFAEC9) used as a full alert-card background with black text
for critical states (Variant B), plus red for the compact "critical" badge
and dot. This supersedes the earlier navy/blue palette that was pulled from
alpaca_paper_trading_specifications_x_quant_x/002_xquantx_aesthetics.txt --
that doc's own changelog admitted it was inherited/not-confirmed-brand.
"""

# --- Core palette (Variant A: normal/monitoring state) ---
BACKGROUND_PRIMARY = "#000000"    # page background
BACKGROUND_SECONDARY = "#0c0a0a"  # sidebar, header
BACKGROUND_CARD = "#170f11"       # data cards, chart panels -- near-black with a warm tint
TEXT_PRIMARY = "#f2ecec"
TEXT_SECONDARY = "#9a8489"        # captions, footnotes, source annotations, muted mauve-gray
GRID_LINE = "#2a2024"
BORDER = "#332628"                # panel borders, dividers
ZERO_LINE = "#4a3a3d"             # signal-chart zero reference line

SERIES_EQUITY = "#B76E79"         # Variant A accent: equity curve, primary data, series-1
SERIES_BENCHMARK = "#7d5a60"      # dimmer mauve: benchmark / secondary series
SERIES_LONG = "#2DD4BF"           # teal: long positions, positive values
SERIES_NEUTRAL = "#6b7280"

ALERT_WARN = "#f59e0b"            # amber: moderate warnings (10-15% drawdown)
ALERT_CRITICAL = "#FFAEC9"        # Variant B: full alert-card bg -- REDUCE/BLOCK/FLATTEN, drawdown >15%, R04/R07 only
ALERT_BADGE = "#e11d48"           # compact red badge/dot for header-level "CRITICAL" state

REGIME_BULL = "#2DD4BF"           # teal (matches "Active"/positive in the mockup)
REGIME_BEAR = "#e11d48"
REGIME_NEUTRAL = "#6b7280"

# --- Typography ---
FONT_MONO = "'JetBrains Mono', 'Roboto Mono', monospace"   # all numeric values
FONT_SANS = "'Inter', 'Roboto', sans-serif"                 # labels, titles, headers
GOOGLE_FONTS_HREF = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap"
)

# Per-regime color mapping
REGIME_COLORS = {
    "R01": REGIME_BULL, "R02": REGIME_BULL, "R11": REGIME_BULL,
    "R03": ALERT_WARN, "R04": ALERT_CRITICAL,
    "R05": REGIME_BEAR, "R07": ALERT_CRITICAL,
    "R06": REGIME_NEUTRAL, "R08": REGIME_NEUTRAL, "R09": REGIME_NEUTRAL,
    "R10": REGIME_NEUTRAL, "R12": REGIME_NEUTRAL,
}

# Verdicts that must render with the alert-critical token (REDUCE/BLOCK/FLATTEN)
ALERT_VERDICTS = {"BLOCK", "FLATTEN", "REDUCE"}

VERDICT_ROW_COLOR = {
    "BLOCK": ALERT_BADGE,
    "FLATTEN": ALERT_CRITICAL,
    "REDUCE": "#fb923c",  # light orange
    "ALLOW": BACKGROUND_CARD,
}

# Text color per verdict row, matched to VERDICT_ROW_COLOR's background lightness
# (ALERT_BADGE and ALLOW are dark -> light text; FLATTEN and REDUCE are light -> dark text)
VERDICT_TEXT_COLOR = {
    "BLOCK": TEXT_PRIMARY,
    "FLATTEN": BACKGROUND_PRIMARY,
    "REDUCE": BACKGROUND_PRIMARY,
    "ALLOW": TEXT_PRIMARY,
}

ORDER_STATUS_COLOR = {
    "filled": REGIME_BULL,
    "cancelled": REGIME_NEUTRAL,
    "new": SERIES_EQUITY,
    "partially_filled": ALERT_WARN,
}

RULESET_VERSION = "xquantx-2026.08"

PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=BACKGROUND_CARD,
    plot_bgcolor=BACKGROUND_CARD,
    font=dict(family=FONT_SANS, color=TEXT_PRIMARY, size=12),
    title=dict(font=dict(family=FONT_SANS, size=14, color=TEXT_PRIMARY)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY, size=11)),
    xaxis=dict(gridcolor=GRID_LINE, zerolinecolor=GRID_LINE, showgrid=True,
               tickfont=dict(family=FONT_SANS, size=11)),
    yaxis=dict(gridcolor=GRID_LINE, zerolinecolor=GRID_LINE, showgrid=True,
               tickfont=dict(family=FONT_SANS, size=11)),
    margin=dict(l=55, r=30, t=60, b=45),
)

CARD_STYLE = {
    "backgroundColor": BACKGROUND_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
    "padding": "4px",
    "flex": "1",
}
