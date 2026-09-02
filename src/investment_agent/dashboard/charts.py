"""X Quant X dashboard chart builders.

WHAT
====
Builds the seven Plotly chart types required by
alpaca_paper_trading_specifications_x_quant_x/022_xquantx_visualization_dev_standards.txt
Section 1.6: equity curve, drawdown waterfall, regime probability bar,
correlation heatmap, signal score time series, portfolio weights treemap, and
the risk-gate trigger log table. Styling follows 002_xquantx_aesthetics.txt
(color tokens, Inter/JetBrains Mono typography, top-horizontal legends, grid
rules) -- see colors.py for the token definitions and the note on the
unresolved brand-submodule question.

WHY
===
Kept separate from layout.py and app.py so chart logic (data -> Figure) stays
independently testable without needing a running Dash server, per
005_xquantx_coding_standards.txt's "no trading logic in dashboard/" boundary
-- this module contains no trading logic, only presentation of data already
computed elsewhere.

HOW
===
Every builder takes plain Python data (lists/dicts from data_loader.py) and
returns a plotly.graph_objects.Figure. Every builder degrades gracefully to
an "empty state" figure with a message instead of raising when given no data,
since a freshly-started paper session may have zero trade history.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from . import colors


def _base_layout(title: str, **overrides) -> dict:
    """Merge chart-specific overrides onto the shared dark-theme defaults.

    Kept as a function (rather than unpacking colors.PLOTLY_LAYOUT_DEFAULTS
    directly into update_layout) so a chart can override xaxis/yaxis/legend
    without losing the rest of that sub-dict's styling, and so `title` never
    collides with the shared defaults.
    """
    layout = dict(colors.PLOTLY_LAYOUT_DEFAULTS)
    layout["title"] = dict(text=title, font=dict(family=colors.FONT_SANS, size=14, color=colors.TEXT_PRIMARY))
    for key, value in overrides.items():
        if key in ("xaxis", "yaxis", "legend") and isinstance(value, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    return layout


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(family=colors.FONT_SANS, color=colors.TEXT_SECONDARY, size=14),
    )
    fig.update_layout(**_base_layout(""))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _source_annotation(fig: go.Figure, session_id: str, mode: str) -> go.Figure:
    fig.add_annotation(
        text=f"X Quant X | {session_id} | {mode}",
        xref="paper", yref="paper",
        x=1.0, y=-0.32,
        showarrow=False,
        font=dict(family=colors.FONT_SANS, color=colors.TEXT_SECONDARY, size=10),
        align="right",
    )
    return fig


def _table_header_font() -> dict:
    return dict(family=colors.FONT_SANS, size=10, color=colors.TEXT_PRIMARY)


def _contrasting_text(bg_hex: str) -> str:
    """Pick black or white text for a given background hex, by relative luminance.

    Row fill colors span the whole palette (dark card tokens, bright pink
    alerts, mid-tone reds/mauves/grays) so a single fixed table-cell text
    color isn't safe -- this picks per-row instead of guessing.
    """
    h = bg_hex.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return colors.TEXT_PRIMARY if luminance < 0.4 else colors.BACKGROUND_PRIMARY


def _table_cell_font(text_colors: List[str]) -> dict:
    return dict(family=colors.FONT_MONO, size=11, color=[text_colors])


# ---------------------------------------------------------------------------
# 1.6.1 Equity Curve
# ---------------------------------------------------------------------------

def build_equity_curve_chart(
    equity_curve: List[Dict[str, Any]],
    session_id: str = "n/a",
    mode: str = "PAPER",
    source: str = "strategy",
) -> go.Figure:
    """Strategy-side analytical equity curve.

    The ``source`` argument is rendered into the title so users can tell
    this apart from broker-side authoritative equity. The two numbers
    will diverge on a real-money session because (a) this curve is
    built from ``TradeExperience.pnl`` deltas, while (b) the broker
    reports total account equity including fees, dividends, options
    P&L, and intraday mark-to-market that the strategy log does not
    capture. They will match on a paper account with no fees and a
    closed ledger.
    """
    if not equity_curve:
        return _empty_figure("No trade history yet")

    timestamps = [row["timestamp"] for row in equity_curve]
    equity = [row["equity"] for row in equity_curve]
    peak = equity_curve[-1]["peak"]
    current_drawdown = equity_curve[-1]["drawdown_pct"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=equity, mode="lines", name="Strategy equity",
        line=dict(color=colors.SERIES_EQUITY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=[equity_curve[0]["equity"]] * len(timestamps),
        mode="lines", name="Baseline", line=dict(color=colors.SERIES_BENCHMARK, dash="dash", width=1),
    ))
    fig.add_hline(y=peak, line=dict(color=colors.TEXT_SECONDARY, dash="dot", width=1),
                  annotation_text=f"Peak ${peak:,.2f}", annotation_font_color=colors.TEXT_SECONDARY,
                  annotation_font_family=colors.FONT_MONO)
    fig.update_layout(**_base_layout(
        f"Equity Curve [{source}] (drawdown {current_drawdown:.1%})",
        xaxis_title="Time", yaxis_title="Equity ($)",
    ))
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# 1.6.2 Drawdown Waterfall
# ---------------------------------------------------------------------------

def build_drawdown_waterfall_chart(
    equity_curve: List[Dict[str, Any]],
    session_id: str = "n/a",
    mode: str = "PAPER",
    flatten_pct: float = 0.15,
    reduce_pct: float = 0.10,
) -> go.Figure:
    """Drawdown waterfall with authoritative FLATTEN / REDUCE thresholds.

    The ``flatten_pct`` and ``reduce_pct`` are the canonical
    ``DRAWDOWN_FLATTEN_PCT`` / ``DRAWDOWN_REDUCE_PCT`` from
    ``capital_gate``. We render them as percent in the chart so the
    axis lines stay at ``-15`` / ``-10`` for the canonical default but
    move to whatever the live config says when those are overridden.
    """
    if not equity_curve:
        return _empty_figure("No trade history yet")

    timestamps = [row["timestamp"] for row in equity_curve]
    drawdown_pct = [row["drawdown_pct"] * 100 for row in equity_curve]

    flatten_pct_disp = -abs(flatten_pct) * 100
    reduce_pct_disp = -abs(reduce_pct) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=drawdown_pct, mode="lines", fill="tozeroy",
        line=dict(color=colors.SERIES_EQUITY),
        fillcolor="rgba(183,110,121,0.18)",
        name="Drawdown %",
    ))
    fig.add_hline(y=flatten_pct_disp, line=dict(color=colors.ALERT_CRITICAL, dash="dash", width=1),
                  annotation_text=f"FLATTEN {flatten_pct_disp:.0f}%",
                  annotation_font_color=colors.ALERT_CRITICAL,
                  annotation_font_family=colors.FONT_MONO)
    fig.add_hline(y=reduce_pct_disp, line=dict(color=colors.ALERT_WARN, dash="dot", width=1),
                  annotation_text=f"Warn {reduce_pct_disp:.0f}%",
                  annotation_font_color=colors.ALERT_WARN,
                  annotation_font_family=colors.FONT_MONO)
    fig.update_layout(**_base_layout(
        "Drawdown from Peak (strategy-side)",
        xaxis_title="Time", yaxis_title="Drawdown (%)",
    ))
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# 1.6.3 Regime Probability Bar (current bar) + history timeline
# ---------------------------------------------------------------------------

def build_regime_probability_chart(history: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not history:
        return _empty_figure("No regime data yet")

    latest = history[-1]
    probs: Dict[str, float] = latest.get("regime_probabilities", {}) or {}
    if not probs:
        return _empty_figure("No regime data yet")

    regimes = sorted(probs.keys())
    values = [probs[r] for r in regimes]
    bar_colors = [colors.REGIME_COLORS.get(r, colors.REGIME_NEUTRAL) for r in regimes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=regimes, orientation="h",
        marker=dict(color=bar_colors),
        text=[f"{v:.1%}" for v in values], textposition="outside",
        textfont=dict(family=colors.FONT_MONO, size=11, color=colors.TEXT_PRIMARY),
    ))
    fig.update_layout(**_base_layout(
        f"Regime Probabilities (current bar, most likely: {latest.get('regime', 'n/a')})",
        xaxis_title="Probability", yaxis_title="Regime",
        xaxis=dict(range=[0, 1.15]),
        showlegend=False,
    ))
    return _source_annotation(fig, session_id, mode)


def build_regime_history_timeline(history: List[Dict[str, Any]], last_n: int = 60, session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not history:
        return _empty_figure("No regime history yet")

    recent = history[-last_n:]
    timestamps = [row.get("timestamp") for row in recent]
    regimes = [row.get("regime", "n/a") for row in recent]
    bar_colors = [colors.REGIME_COLORS.get(r, colors.REGIME_NEUTRAL) for r in regimes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=timestamps, y=[1] * len(recent), marker=dict(color=bar_colors),
        text=regimes, hovertext=regimes, hoverinfo="text",
    ))
    fig.update_layout(**_base_layout(f"Regime History (last {len(recent)} bars)", showlegend=False))
    fig.update_yaxes(visible=False)
    return _source_annotation(fig, session_id, mode)


def compute_entropy_gauge_value(history: List[Dict[str, Any]]) -> float:
    if not history:
        return 0.0
    from .data_loader import compute_regime_entropy
    return compute_regime_entropy(history[-1].get("regime_probabilities", {}) or {})


def build_entropy_gauge(history: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    entropy = compute_entropy_gauge_value(history)
    bar_color = colors.ALERT_CRITICAL if entropy > 0.75 else colors.SERIES_EQUITY
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=entropy,
        gauge=dict(
            axis=dict(range=[0, 1], tickcolor=colors.TEXT_SECONDARY, tickfont=dict(family=colors.FONT_MONO)),
            bar=dict(color=bar_color),
            bgcolor=colors.BACKGROUND_CARD,
            borderwidth=1, bordercolor=colors.BORDER,
            threshold=dict(line=dict(color=colors.ALERT_CRITICAL, width=3), value=0.75),
        ),
        number=dict(font=dict(family=colors.FONT_MONO, color=colors.TEXT_PRIMARY)),
        title=dict(text="Regime Entropy (U_t)", font=dict(family=colors.FONT_SANS, color=colors.TEXT_PRIMARY)),
    ))
    fig.update_layout(**_base_layout(""))
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# 1.6.4 Correlation Heatmap
# ---------------------------------------------------------------------------

def build_correlation_heatmap_chart(history: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    symbols = sorted({row.get("symbol") for row in history if row.get("symbol")})
    if len(symbols) < 2:
        return _empty_figure("Need 2+ symbols traded to compute correlation")

    import pandas as pd

    df = pd.DataFrame(history)
    pivot = df.pivot_table(index="timestamp", columns="symbol", values="ensemble_signal", aggfunc="mean")
    corr = pivot.corr(min_periods=1)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdYlGn", zmin=-1, zmax=1,
        colorbar=dict(title="corr", tickfont=dict(family=colors.FONT_MONO, color=colors.TEXT_SECONDARY)),
    ))
    fig.update_layout(**_base_layout("Signal Correlation Heatmap", showlegend=False))
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# 1.6.5 Signal Score Time Series
# ---------------------------------------------------------------------------

def build_signal_score_chart(history: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not history:
        return _empty_figure("No signal history yet")

    timestamps = [row.get("timestamp") for row in history]
    agent_ids = sorted({aid for row in history for aid in (row.get("agent_signals") or {}).keys()})

    fig = go.Figure()
    palette = [colors.SERIES_EQUITY, colors.REGIME_BULL, colors.ALERT_WARN,
               colors.SERIES_BENCHMARK, colors.SERIES_NEUTRAL, "#BA68C8", "#4DD0E1"]
    for i, agent_id in enumerate(agent_ids):
        y = [(row.get("agent_signals") or {}).get(agent_id, 0.0) for row in history]
        fig.add_trace(go.Scatter(
            x=timestamps, y=y, mode="lines", name=agent_id,
            line=dict(color=palette[i % len(palette)], width=1.5),
        ))

    ensemble = [row.get("ensemble_signal", 0.0) for row in history]
    confidence = [row.get("effective_confidence", 0.0) for row in history]
    upper = [e + (1 - c) * 0.5 for e, c in zip(ensemble, confidence)]
    lower = [e - (1 - c) * 0.5 for e, c in zip(ensemble, confidence)]
    fig.add_trace(go.Scatter(x=timestamps + timestamps[::-1], y=upper + lower[::-1],
                              fill="toself", fillcolor="rgba(183,110,121,0.18)",
                              line=dict(color="rgba(0,0,0,0)"), name="Ensemble confidence band",
                              showlegend=True, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=timestamps, y=ensemble, mode="lines", name="Ensemble (aggregate)",
                              line=dict(color=colors.TEXT_PRIMARY, width=3)))
    fig.add_hline(y=0, line=dict(color=colors.ZERO_LINE, width=1))
    fig.update_layout(**_base_layout(
        "Signal Scores (per agent + ensemble)",
        xaxis_title="Time", yaxis_title="Signal score [-1, +1]",
        yaxis=dict(range=[-1, 1]),
    ))
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# 1.6.6 Portfolio Weights Treemap
# ---------------------------------------------------------------------------

def build_portfolio_treemap(positions: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not positions:
        return _empty_figure("No open positions")

    labels, values, treemap_colors, texts = [], [], [], []
    for p in positions:
        weight = abs(p.get("market_value") or 0.0)
        labels.append(p["symbol"])
        values.append(weight if weight > 0 else 1)
        treemap_colors.append(colors.SERIES_LONG if p.get("side") == "long" else colors.REGIME_BEAR)
        equity_val = p.get("market_value")
        texts.append(f"{p['symbol']}<br>${equity_val:,.2f}" if equity_val is not None else p["symbol"])

    text_colors = [_contrasting_text(c) for c in treemap_colors]
    fig = go.Figure(go.Treemap(
        labels=labels, parents=[""] * len(labels), values=values,
        marker=dict(colors=treemap_colors, line=dict(color=colors.BACKGROUND_CARD, width=3)),
        text=texts, textinfo="text",
        textfont=dict(family=colors.FONT_SANS, size=13, color=text_colors),
        pathbar=dict(visible=False),
    ))
    fig.update_layout(**_base_layout("Portfolio Weights", showlegend=False))
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# 1.6.7 Risk-Gate Trigger Log Table  (also reused for P04 / P05 tables)
# ---------------------------------------------------------------------------

def build_risk_gate_table(risk_log: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not risk_log:
        return _empty_figure("No risk-gate events yet")

    rows = risk_log[-200:][::-1]  # most recent first
    row_colors = [colors.VERDICT_ROW_COLOR.get(r["verdict"], colors.BACKGROUND_CARD) for r in rows]
    # Row backgrounds mix dark (ALLOW, BLOCK) and light (FLATTEN, REDUCE) tokens
    # in the same column, so font color has to follow fill color per row.
    text_colors = [_contrasting_text(c) for c in row_colors]

    fig = go.Figure(go.Table(
        columnwidth=[140, 90, 80, 70, 110, 80],
        header=dict(
            values=["Timestamp", "Rule ID", "Verdict", "Symbol", "Measured Value", "Threshold"],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [r["timestamp"] for r in rows],
                [r["rule_id"] for r in rows],
                [r["verdict"] for r in rows],
                [r["symbol"] for r in rows],
                [f"{r['measured_value']:.4f}" if r["measured_value"] is not None else "n/a" for r in rows],
                [r["threshold"] for r in rows],
            ],
            fill_color=[row_colors],
            font=_table_cell_font(text_colors),
            align="left", height=26,
        ),
    ))
    fig.update_layout(**_base_layout("Risk-Gate Trigger Log", showlegend=False))
    return _source_annotation(fig, session_id, mode)


def build_position_table(positions: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not positions:
        return _empty_figure("No open positions")

    fig = go.Figure(go.Table(
        header=dict(
            values=["Symbol", "Side", "Qty", "Avg Entry", "Current Price", "Unrealized P&L ($)", "Unrealized P&L (%)"],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [p["symbol"] for p in positions],
                [p["side"] for p in positions],
                [p["qty"] for p in positions],
                [p["avg_entry_price"] for p in positions],
                [p["current_price"] for p in positions],
                [p["unrealized_pl"] for p in positions],
                [f"{(p['unrealized_plpc'] or 0) * 100:.2f}%" if p["unrealized_plpc"] is not None else "n/a" for p in positions],
            ],
            fill_color=[[
                colors.SERIES_LONG if (p.get("unrealized_pl") or 0) >= 0 else colors.REGIME_BEAR
                for p in positions
            ]],
            font=_table_cell_font([
                _contrasting_text(colors.SERIES_LONG if (p.get("unrealized_pl") or 0) >= 0 else colors.REGIME_BEAR)
                for p in positions
            ]),
            align="left", height=26,
        ),
    ))
    fig.update_layout(**_base_layout("Position Detail", showlegend=False))
    return _source_annotation(fig, session_id, mode)


def build_order_history_table(orders: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not orders:
        return _empty_figure("No orders yet")

    row_colors = [colors.ORDER_STATUS_COLOR.get(o.get("status"), colors.BACKGROUND_CARD) for o in orders]
    text_colors = [_contrasting_text(c) for c in row_colors]

    fig = go.Figure(go.Table(
        header=dict(
            values=["Timestamp", "Order ID", "Symbol", "Side", "Type", "Qty", "Filled Qty", "Fill Price", "Status"],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [o["timestamp"] for o in orders],
                [o["order_id"] for o in orders],
                [o["symbol"] for o in orders],
                [o["side"] for o in orders],
                [o["type"] for o in orders],
                [o["qty"] for o in orders],
                [o["filled_qty"] for o in orders],
                [o["filled_avg_price"] for o in orders],
                [o["status"] for o in orders],
            ],
            fill_color=[row_colors],
            font=_table_cell_font(text_colors),
            align="left", height=26,
        ),
    ))
    fig.update_layout(**_base_layout("Order History", showlegend=False))
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# Control-room panel builders (single scrolling layout, no P01-P06 tabs)
# ---------------------------------------------------------------------------

def build_seven_state_soc_chart(charges: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    """Horizontal bar chart of the seven state-of-charge values in [0, 1]."""
    if not charges:
        return _empty_figure("No state-of-charge data yet")
    labels = [c.get("label", "?") for c in charges]
    values = [float(c.get("value", 0.0) or 0.0) for c in charges]
    bar_colors = [
        colors.SERIES_LONG if v >= 0.66 else colors.ALERT_WARN if v >= 0.33 else colors.ALERT_CRITICAL
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=bar_colors),
        text=[f"{v:.2f}" for v in values], textposition="outside",
        textfont=dict(family=colors.FONT_MONO, size=11, color=colors.TEXT_PRIMARY),
    ))
    fig.update_layout(**_base_layout(
        "7-State Capital Gate: State-of-Charge",
        xaxis_title="Charge", yaxis_title="State",
        xaxis=dict(range=[0, 1.15]),
        showlegend=False,
    ))
    return _source_annotation(fig, session_id, mode)


def build_seven_agents_table(agents: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    """Per-agent signal / confidence / weight / status table for the latest cycle.

    Each row surfaces the full channels that the orchestrator records
    in ``agent_outputs_full`` (signal, confidence, uncertainty, doubt,
    p_bull, p_bear, decision time, kalman noise, weight, reputation
    alpha/beta) so an operator can audit the ensemble at a glance.
    Legacy rows that only carry the four basic fields are rendered
    with a status of ``ok (legacy)`` and blank extended columns.
    """
    if not agents:
        return _empty_figure("No agent signals yet")
    rows = agents

    def _f(d, key, default=None):
        v = d.get(key)
        if v is None:
            return default
        return v

    row_colors = [
        colors.SERIES_LONG if _f(r, "signal", 0.0) >= 0 else colors.REGIME_BEAR
        for r in rows
    ]
    text_colors = [_contrasting_text(c) for c in row_colors]

    def _sig(r):
        s = _f(r, "signal")
        return f"{s:+.2f}" if isinstance(s, (int, float)) else "n/a"

    def _conf(r):
        c = _f(r, "confidence")
        return f"{c:.2f}" if isinstance(c, (int, float)) else "n/a"

    def _unc(r):
        u = _f(r, "uncertainty")
        return f"{u:.2f}" if isinstance(u, (int, float)) else "—"

    def _doubt(r):
        d_ = _f(r, "doubt")
        return f"{d_:.2f}" if isinstance(d_, (int, float)) else "—"

    def _pbull(r):
        v = _f(r, "p_bull")
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    def _pbear(r):
        v = _f(r, "p_bear")
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    def _dt(r):
        v = _f(r, "decision_time_ms")
        return f"{int(v)} ms" if isinstance(v, (int, float)) else "—"

    def _w(r):
        w = _f(r, "weight")
        return f"{w:.2f}" if isinstance(w, (int, float)) else "n/a"

    def _rep(r):
        a, b = _f(r, "alpha"), _f(r, "beta")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return f"α{a:.1f}/β{b:.1f}"
        return "—"

    fig = go.Figure(go.Table(
        columnwidth=[150, 65, 60, 60, 65, 65, 65, 55, 65, 100],
        header=dict(
            values=[
                "Agent", "Signal", "Conf", "Unc", "Doubt",
                "p_Bull", "p_Bear", "Δt", "Weight", "Reputation",
            ],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [_f(r, "agent_id", "") for r in rows],
                [_sig(r) for r in rows],
                [_conf(r) for r in rows],
                [_unc(r) for r in rows],
                [_doubt(r) for r in rows],
                [_pbull(r) for r in rows],
                [_pbear(r) for r in rows],
                [_dt(r) for r in rows],
                [_w(r) for r in rows],
                [_rep(r) for r in rows],
            ],
            fill_color=[row_colors],
            font=_table_cell_font(text_colors),
            align="left", height=24,
        ),
    ))
    fig.update_layout(**_base_layout(
        "7 Specialist Agents (full per-agent channels, latest cycle)",
        showlegend=False,
    ))
    return _source_annotation(fig, session_id, mode)


def build_kalman_chart(kalman: Dict[str, Any], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    """Bar chart of prior / observation / posterior with the gain as a gauge.

    The ``kalman`` payload can come from either the authoritative path
    (``run_cycle``'s ``TradeExperience`` audit fields) or a legacy
    reconstruction. The ``posterior_authoritative`` flag indicates
    whether the posterior was the real state-gated
    ``capital_gate.effective_cap`` (preferred) or a backfill from the
    agent side. We surface this in the title and adjust the posterior
    bar color to match.
    """
    if not kalman:
        return _empty_figure("No Kalman data yet")
    kg = float(kalman.get("kalman_gain", 0.0) or 0.0)
    prior = float(kalman.get("prior_confidence", 0.0) or 0.0)
    obs = float(kalman.get("market_observation", 0.0) or 0.0)
    posterior = float(kalman.get("posterior_estimate", 0.0) or 0.0)
    cats = ["Prior", "Observation", "Posterior"]
    vals = [prior, obs, posterior]
    is_authoritative = bool(kalman.get("posterior_authoritative", False))
    posterior_color = colors.SERIES_LONG if is_authoritative else colors.ALERT_WARN
    bar_colors = [colors.SERIES_BENCHMARK, colors.SERIES_EQUITY, posterior_color]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cats, y=vals, marker=dict(color=bar_colors),
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
        textfont=dict(family=colors.FONT_MONO, size=11, color=colors.TEXT_PRIMARY),
    ))
    pos_label = "authoritative (state-gated)" if is_authoritative else "reconstructed (legacy)"
    fig.update_layout(**_base_layout(
        f"Investment Kalman: prior → posterior [{pos_label}] (K = {kg:.2f})",
        yaxis_title="Belief",
        yaxis=dict(range=[-1, 1]),
        showlegend=False,
    ))
    return _source_annotation(fig, session_id, mode)


def build_regime_panel_chart(regime_card: Dict[str, Any], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    """Current-regime probability bar chart."""
    if not regime_card or not regime_card.get("regime"):
        return _empty_figure("No regime data yet")
    probs = regime_card.get("probabilities", {}) or {}
    if not probs:
        return _empty_figure("No regime data yet")
    regimes = sorted(probs.keys())
    values = [float(probs.get(r, 0.0) or 0.0) for r in regimes]
    bar_colors = [colors.REGIME_COLORS.get(r, colors.REGIME_NEUTRAL) for r in regimes]
    fig = go.Figure(go.Bar(
        x=values, y=regimes, orientation="h",
        marker=dict(color=bar_colors),
        text=[f"{v:.1%}" for v in values], textposition="outside",
        textfont=dict(family=colors.FONT_MONO, size=11, color=colors.TEXT_PRIMARY),
    ))
    fig.update_layout(**_base_layout(
        f"Current Regime: {regime_card.get('regime', 'n/a')} "
        f"(top prob {regime_card.get('top_probability', 0.0):.1%})",
        xaxis_title="Probability", yaxis_title="Regime",
        xaxis=dict(range=[0, 1.15]),
        showlegend=False,
    ))
    return _source_annotation(fig, session_id, mode)


def build_llm_providers_table(rows: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    """Per-provider health table: status, model, latency, tokens, success/fail counts."""
    if not rows:
        return _empty_figure("No LLM calls recorded yet")
    ordered = sorted(rows, key=lambda r: r.get("provider_id", ""))
    row_colors = [
        colors.SERIES_LONG if r.get("last_status") == "ok" else colors.ALERT_WARN
        for r in ordered
    ]
    text_colors = [_contrasting_text(c) for c in row_colors]
    def _fmt_model(m: str) -> str:
        s = m.split("/")[-1] if "/" in m else m
        return s[:25] + "..." if len(s) > 28 else s

    fig = go.Figure(go.Table(
        columnwidth=[130, 200, 70, 90, 90, 70, 70],
        header=dict(
            values=["Provider", "Model", "Status", "Last latency", "Last tokens", "Success", "Fail"],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [r.get("provider_id", "") for r in ordered],
                [_fmt_model(r.get("model", "") or "") for r in ordered],
                [r.get("last_status", "ok") for r in ordered],
                [f"{r.get('last_latency_ms', 0.0):.0f} ms" for r in ordered],
                [str(r.get("last_tokens", 0)) for r in ordered],
                [str(r.get("success_calls", 0)) for r in ordered],
                [str(r.get("failure_calls", 0)) for r in ordered],
            ],
            fill_color=[row_colors],
            font=_table_cell_font(text_colors),
            align="left", height=24,
        ),
    ))
    fig.update_layout(**_base_layout("LLM Providers (Featherless, last call per provider)", showlegend=False))
    return _source_annotation(fig, session_id, mode)


def build_options_table(
    contracts: List[Dict[str, Any]],
    session_id: str = "n/a",
    mode: str = "PAPER",
    error: Optional[str] = None,
) -> go.Figure:
    """Options activity table.

    ``contracts`` is the order-history list produced by
    ``data_loader.get_recent_options_activity`` which is the
    authoritative path: we filter the broker's real ``/orders`` feed
    for OCC-format symbols. ``error`` is shown in the empty state when
    the broker call failed so an operator doesn't mistake 'no options'
    for 'no data'.
    """
    if error:
        return _empty_figure(f"Options broker call failed: {error}")
    if not contracts:
        return _empty_figure("No options activity yet (broker /orders filter returned 0 option-shaped orders)")
    fig = go.Figure(go.Table(
        columnwidth=[90, 220, 60, 80, 90, 80, 100],
        header=dict(
            values=[
                "Underlying", "Contract", "Side", "Type", "Qty", "Filled", "Status",
            ],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [c.get("underlying", "") for c in contracts],
                [c.get("symbol", "") for c in contracts],
                [c.get("side", "") for c in contracts],
                [c.get("type", "") for c in contracts],
                [c.get("qty", "") for c in contracts],
                [c.get("filled_qty", "") for c in contracts],
                [c.get("status", "") for c in contracts],
            ],
            fill_color=[[colors.BACKGROUND_CARD] * len(contracts)],
            font=_table_cell_font([colors.TEXT_PRIMARY] * len(contracts)),
            align="left", height=26,
        ),
    ))
    fig.update_layout(**_base_layout("Options Activity (real broker /orders filter, OCC symbols)", showlegend=False))
    return _source_annotation(fig, session_id, mode)


def build_trade_outcome_table(rows: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    """Per-agent accuracy table for the last N closed trades."""
    if not rows:
        return _empty_figure("No closed trades yet")
    ordered = sorted(rows, key=lambda r: r.get("agent_id", ""))
    row_colors = [
        colors.SERIES_LONG if r.get("accuracy", 0.0) >= 0.66 else
        colors.ALERT_WARN if r.get("accuracy", 0.0) >= 0.5 else
        colors.REGIME_BEAR
        for r in ordered
    ]
    text_colors = [_contrasting_text(c) for c in row_colors]
    fig = go.Figure(go.Table(
        columnwidth=[200, 90, 100, 100],
        header=dict(
            values=["Agent", "Correct", "Incorrect", "Accuracy"],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [r.get("agent_id", "") for r in ordered],
                [str(r.get("correct", 0)) for r in ordered],
                [str(r.get("incorrect", 0)) for r in ordered],
                [f"{r.get('accuracy', 0.0):.1%}" for r in ordered],
            ],
            fill_color=[row_colors],
            font=_table_cell_font(text_colors),
            align="left", height=26,
        ),
    ))
    fig.update_layout(**_base_layout("Trade Outcome Learning (last 50 closed trades)", showlegend=False))
    return _source_annotation(fig, session_id, mode)


def build_reputation_table(rows: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    """Per-agent Beta(α, β) reputation table."""
    if not rows:
        return _empty_figure("No reputation data yet")
    ordered = sorted(rows, key=lambda r: r.get("agent_id", ""))
    fig = go.Figure(go.Table(
        columnwidth=[220, 90, 90, 110, 110],
        header=dict(
            values=["Agent", "α", "β", "Weight", "Closed trades"],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [r.get("agent_id", "") for r in ordered],
                [f"{r.get('alpha', 1.0):.1f}" for r in ordered],
                [f"{r.get('beta', 1.0):.1f}" for r in ordered],
                [f"{r.get('weight', 0.5):.2f}" for r in ordered],
                [str(r.get("closed_trades", 0)) for r in ordered],
            ],
            fill_color=[[colors.BACKGROUND_CARD] * len(ordered)],
            font=_table_cell_font([colors.TEXT_PRIMARY] * len(ordered)),
            align="left", height=26,
        ),
    ))
    fig.update_layout(**_base_layout("Agent Reputation (Beta-Bernoulli priors)", showlegend=False))
    return _source_annotation(fig, session_id, mode)


def build_decision_waterfall(steps: List[Dict[str, Any]]) -> go.Figure:
    """Vertical 'Why did X Quant X trade?' stepper rendered as a Plotly table."""
    if not steps:
        return _empty_figure("No decision recorded yet")
    status_color = {
        "pass": colors.SERIES_LONG,
        "warn": colors.ALERT_WARN,
        "fail": colors.ALERT_BADGE,
        "info": colors.BACKGROUND_SECONDARY,
    }
    row_colors = [status_color.get(s.get("status", "info"), colors.BACKGROUND_CARD) for s in steps]
    text_colors = [_contrasting_text(c) for c in row_colors]
    fig = go.Figure(go.Table(
        columnwidth=[60, 240, 220],
        header=dict(
            values=["#", "Stage", "Value"],
            fill_color=colors.BACKGROUND_SECONDARY, font=_table_header_font(),
            align="left", height=28,
        ),
        cells=dict(
            values=[
                [str(i + 1) for i, _ in enumerate(steps)],
                [s.get("stage", "") for s in steps],
                [s.get("value", "") for s in steps],
            ],
            fill_color=[row_colors],
            font=_table_cell_font(text_colors),
            align="left", height=26,
        ),
    ))
    fig.update_layout(**_base_layout("", showlegend=False))
    return fig
