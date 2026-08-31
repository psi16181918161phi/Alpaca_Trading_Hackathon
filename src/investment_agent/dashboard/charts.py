"""X Quant X dashboard chart builders.

WHAT
====
Builds the seven Plotly chart types required by
alpaca_paper_trading_specifications_x_quant_x/022_xquantx_visualization_dev_standards.txt
Section 1.6: equity curve, drawdown waterfall, regime probability bar,
correlation heatmap, signal score time series, portfolio weights treemap, and
the risk-gate trigger log table.

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


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=colors.TEXT_PRIMARY, size=14),
    )
    fig.update_layout(**colors.PLOTLY_LAYOUT_DEFAULTS)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _source_annotation(fig: go.Figure, session_id: str, mode: str) -> go.Figure:
    fig.add_annotation(
        text=f"X Quant X | {session_id} | {mode}",
        xref="paper", yref="paper",
        x=1.0, y=-0.18,
        showarrow=False,
        font=dict(color=colors.GRID_LINE, size=10),
        align="right",
    )
    return fig


# ---------------------------------------------------------------------------
# 1.6.1 Equity Curve
# ---------------------------------------------------------------------------

def build_equity_curve_chart(equity_curve: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not equity_curve:
        return _empty_figure("No trade history yet")

    timestamps = [row["timestamp"] for row in equity_curve]
    equity = [row["equity"] for row in equity_curve]
    peak = equity_curve[-1]["peak"]
    current_drawdown = equity_curve[-1]["drawdown_pct"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=equity, mode="lines", name="Portfolio equity",
        line=dict(color=colors.SERIES_EQUITY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=[equity_curve[0]["equity"]] * len(timestamps),
        mode="lines", name="Baseline", line=dict(color=colors.SERIES_BENCHMARK, dash="dash"),
    ))
    fig.add_hline(y=peak, line=dict(color=colors.GRID_LINE, dash="dot"),
                  annotation_text=f"Peak ${peak:,.2f}", annotation_font_color=colors.TEXT_PRIMARY)
    fig.update_layout(
        title=f"Equity Curve (drawdown {current_drawdown:.1%})",
        xaxis_title="Time", yaxis_title="Equity ($)",
        **colors.PLOTLY_LAYOUT_DEFAULTS,
    )
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# 1.6.2 Drawdown Waterfall
# ---------------------------------------------------------------------------

def build_drawdown_waterfall_chart(equity_curve: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not equity_curve:
        return _empty_figure("No trade history yet")

    timestamps = [row["timestamp"] for row in equity_curve]
    drawdown_pct = [row["drawdown_pct"] * 100 for row in equity_curve]
    point_colors = []
    for dd in drawdown_pct:
        if dd <= -15:
            point_colors.append(colors.ALERT_CRITICAL)
        elif dd <= -10:
            point_colors.append(colors.ALERT_WARN)
        else:
            point_colors.append(colors.SERIES_EQUITY)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=drawdown_pct, mode="lines", fill="tozeroy",
        line=dict(color=colors.SERIES_EQUITY),
        marker=dict(color=point_colors),
        name="Drawdown %",
    ))
    fig.add_hline(y=-15, line=dict(color=colors.ALERT_CRITICAL, dash="dash"),
                  annotation_text="FLATTEN threshold", annotation_font_color=colors.ALERT_CRITICAL)
    fig.add_hline(y=-10, line=dict(color=colors.ALERT_WARN, dash="dot"),
                  annotation_text="Warn threshold", annotation_font_color=colors.ALERT_WARN)
    fig.update_layout(
        title="Drawdown from Peak",
        xaxis_title="Time", yaxis_title="Drawdown (%)",
        **colors.PLOTLY_LAYOUT_DEFAULTS,
    )
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
    ))
    fig.update_layout(
        title=f"Regime Probabilities (current bar, most likely: {latest.get('regime', 'n/a')})",
        xaxis_title="Probability", yaxis_title="Regime", xaxis=dict(range=[0, 1]),
        **{k: v for k, v in colors.PLOTLY_LAYOUT_DEFAULTS.items() if k != "xaxis"},
    )
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
    fig.update_layout(
        title=f"Regime History (last {len(recent)} bars)",
        showlegend=False,
        **colors.PLOTLY_LAYOUT_DEFAULTS,
    )
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
            axis=dict(range=[0, 1], tickcolor=colors.TEXT_PRIMARY),
            bar=dict(color=bar_color),
            bgcolor=colors.BACKGROUND_PRIMARY,
            borderwidth=1, bordercolor=colors.GRID_LINE,
            threshold=dict(line=dict(color=colors.ALERT_CRITICAL, width=3), value=0.75),
        ),
        number=dict(font=dict(color=colors.TEXT_PRIMARY)),
        title=dict(text="Regime Entropy (U_t)", font=dict(color=colors.TEXT_PRIMARY)),
    ))
    fig.update_layout(**colors.PLOTLY_LAYOUT_DEFAULTS)
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
        colorbar=dict(title="corr"),
    ))
    fig.update_layout(title="Signal Correlation Heatmap", **colors.PLOTLY_LAYOUT_DEFAULTS)
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
               colors.SERIES_BENCHMARK, colors.REGIME_NEUTRAL, "#BA68C8", "#4DD0E1"]
    for i, agent_id in enumerate(agent_ids):
        y = [(row.get("agent_signals") or {}).get(agent_id, 0.0) for row in history]
        fig.add_trace(go.Scatter(
            x=timestamps, y=y, mode="lines", name=agent_id,
            line=dict(color=palette[i % len(palette)]),
        ))

    ensemble = [row.get("ensemble_signal", 0.0) for row in history]
    confidence = [row.get("effective_confidence", 0.0) for row in history]
    upper = [e + (1 - c) * 0.5 for e, c in zip(ensemble, confidence)]
    lower = [e - (1 - c) * 0.5 for e, c in zip(ensemble, confidence)]
    fig.add_trace(go.Scatter(x=timestamps + timestamps[::-1], y=upper + lower[::-1],
                              fill="toself", fillcolor="rgba(126,200,227,0.15)",
                              line=dict(color="rgba(0,0,0,0)"), name="Ensemble confidence band",
                              showlegend=True, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=timestamps, y=ensemble, mode="lines", name="Ensemble (aggregate)",
                              line=dict(color=colors.TEXT_PRIMARY, width=3)))
    fig.add_hline(y=0, line=dict(color=colors.GRID_LINE))
    fig.update_layout(
        title="Signal Scores (per agent + ensemble)",
        xaxis_title="Time", yaxis_title="Signal score [-1, +1]",
        yaxis=dict(range=[-1, 1], gridcolor=colors.GRID_LINE),
        **{k: v for k, v in colors.PLOTLY_LAYOUT_DEFAULTS.items() if k != "yaxis"},
    )
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
        treemap_colors.append(colors.REGIME_BULL if p.get("side") == "long" else colors.REGIME_BEAR)
        equity_val = p.get("market_value")
        texts.append(f"{p['symbol']}<br>${equity_val:,.2f}" if equity_val is not None else p["symbol"])

    fig = go.Figure(go.Treemap(
        labels=labels, parents=[""] * len(labels), values=values,
        marker=dict(colors=treemap_colors), text=texts, textinfo="text",
    ))
    fig.update_layout(title="Portfolio Weights", **colors.PLOTLY_LAYOUT_DEFAULTS)
    return _source_annotation(fig, session_id, mode)


# ---------------------------------------------------------------------------
# 1.6.7 Risk-Gate Trigger Log Table  (also reused for P04 / P05 tables)
# ---------------------------------------------------------------------------

def build_risk_gate_table(risk_log: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not risk_log:
        return _empty_figure("No risk-gate events yet")

    rows = risk_log[-200:][::-1]  # most recent first
    row_colors = [colors.VERDICT_ROW_COLOR.get(r["verdict"], colors.BACKGROUND_PRIMARY) for r in rows]

    fig = go.Figure(go.Table(
        header=dict(
            values=["Timestamp", "Rule ID", "Verdict", "Symbol", "Measured Value", "Threshold"],
            fill_color=colors.GRID_LINE, font=dict(color=colors.TEXT_PRIMARY),
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
            font=dict(color="#1a1a2e"),
        ),
    ))
    fig.update_layout(title="Risk-Gate Trigger Log", **colors.PLOTLY_LAYOUT_DEFAULTS)
    return _source_annotation(fig, session_id, mode)


def build_position_table(positions: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not positions:
        return _empty_figure("No open positions")

    fig = go.Figure(go.Table(
        header=dict(
            values=["Symbol", "Side", "Qty", "Avg Entry", "Current Price", "Unrealized P&L ($)", "Unrealized P&L (%)"],
            fill_color=colors.GRID_LINE, font=dict(color=colors.TEXT_PRIMARY),
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
                colors.REGIME_BULL if (p.get("unrealized_pl") or 0) >= 0 else colors.REGIME_BEAR
                for p in positions
            ]],
            font=dict(color="#1a1a2e"),
        ),
    ))
    fig.update_layout(title="Position Detail", **colors.PLOTLY_LAYOUT_DEFAULTS)
    return _source_annotation(fig, session_id, mode)


def build_order_history_table(orders: List[Dict[str, Any]], session_id: str = "n/a", mode: str = "PAPER") -> go.Figure:
    if not orders:
        return _empty_figure("No orders yet")

    row_colors = [colors.ORDER_STATUS_COLOR.get(o.get("status"), colors.BACKGROUND_PRIMARY) for o in orders]

    fig = go.Figure(go.Table(
        header=dict(
            values=["Timestamp", "Order ID", "Symbol", "Side", "Type", "Qty", "Filled Qty", "Fill Price", "Status"],
            fill_color=colors.GRID_LINE, font=dict(color=colors.TEXT_PRIMARY),
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
            font=dict(color="#1a1a2e"),
        ),
    ))
    fig.update_layout(title="Order History", **colors.PLOTLY_LAYOUT_DEFAULTS)
    return _source_annotation(fig, session_id, mode)
