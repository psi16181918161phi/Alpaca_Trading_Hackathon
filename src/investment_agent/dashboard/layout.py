"""X Quant X dashboard layout.

WHAT
====
Builds the single "control-room" layout: header, top metrics row,
equity / P&L chart, the AI decision card, the seven-agent table, the
seven-state capital-gate chart, the Kalman estimation chart, the
regime panel, the risk gates + circuit-breaker panel, the LLM
provider health table, the Alpaca account / orders / options section,
the trade-outcome learning + reputation tables, and the "Why did X
Quant X trade?" decision waterfall.

WHY
===
Monitoring-only by construction: nothing in this module can submit an
order, modify a position, or change risk rules. The dashboard never
imports the Alpaca TradingClient; data access goes through
``data_loader.py`` which only calls the read-only helpers exposed by
``investment_agent.execution.execution``.

HOW
====
Plain ``dash.html`` / ``dash.dcc`` components only. Layout builders
take pre-computed data (from ``data_loader.py``) and pre-built Plotly
figures (from ``charts.py``); this module does no data fetching or
chart math of its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html

from . import colors


PAGE_STYLE = {
    "backgroundColor": colors.BACKGROUND_PRIMARY,
    "minHeight": "100vh",
    "color": colors.TEXT_PRIMARY,
    "fontFamily": colors.FONT_SANS,
}

CARD = {
    "backgroundColor": colors.BACKGROUND_CARD,
    "border": f"1px solid {colors.BORDER}",
    "borderRadius": "10px",
    "overflow": "hidden",
}

HEADER_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "padding": "14px 28px",
    "backgroundColor": colors.BACKGROUND_SECONDARY,
    "borderBottom": f"1px solid {colors.BORDER}",
    "color": colors.TEXT_PRIMARY,
    "fontFamily": colors.FONT_SANS,
}

SECTION_STYLE = {
    "backgroundColor": colors.BACKGROUND_CARD,
    "border": f"1px solid {colors.BORDER}",
    "borderRadius": "10px",
    "padding": "14px 18px",
    "marginBottom": "14px",
}

SECTION_TITLE_STYLE = {
    "fontSize": "12px",
    "color": colors.TEXT_SECONDARY,
    "letterSpacing": "1px",
    "fontWeight": "700",
    "marginBottom": "10px",
    "textTransform": "uppercase",
}

KPI_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(6, minmax(0, 1fr))",
    "gap": "10px",
    "marginBottom": "14px",
}

KPI_LABEL_STYLE = {
    "fontSize": "10px",
    "color": colors.TEXT_SECONDARY,
    "letterSpacing": "0.5px",
    "fontWeight": "600",
    "textTransform": "uppercase",
}

KPI_VALUE_STYLE = {
    "fontSize": "20px",
    "fontFamily": colors.FONT_MONO,
    "color": colors.TEXT_PRIMARY,
    "fontWeight": "700",
    "marginTop": "2px",
}


def build_header(session_id, mode="PAPER"):
    mode_bg = colors.SERIES_BENCHMARK
    mode_fg = colors.TEXT_PRIMARY
    if mode == "PAPER":
        mode_bg = colors.SERIES_BENCHMARK
    elif mode == "BACKTEST":
        mode_bg = colors.SERIES_EQUITY
    elif mode == "LIVE":
        mode_bg = colors.ALERT_CRITICAL
        mode_fg = colors.BACKGROUND_PRIMARY
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Span("X QUANT X", style={"fontWeight": "800", "letterSpacing": "2px", "marginRight": "20px", "fontSize": "15px"}),
                    html.Span("AI INVESTMENT SYSTEM", style={"marginRight": "20px", "color": colors.TEXT_SECONDARY, "fontSize": "11px", "letterSpacing": "2px", "fontWeight": "600"}),
                    html.Span(mode, style={"backgroundColor": mode_bg, "color": mode_fg, "padding": "3px 12px", "borderRadius": "5px", "marginRight": "18px", "fontWeight": "700", "fontSize": "11px", "letterSpacing": "1px"}),
                    html.Span("session: " + str(session_id), style={"marginRight": "18px", "color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
                    html.Span("ruleset: " + str(colors.RULESET_VERSION), style={"color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                children=[
                    html.Span(id="dashboard-elapsed", style={"marginRight": "18px", "color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
                    html.Span(id="dashboard-last-refresh", style={"color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
                ]
            ),
        ],
        style=HEADER_STYLE,
    )


def kpi(label, value, alert=False):
    border = colors.ALERT_CRITICAL if alert else colors.BORDER
    value_color = colors.ALERT_CRITICAL if alert else colors.TEXT_PRIMARY
    return html.Div(
        children=[
            html.Div(label, style=KPI_LABEL_STYLE),
            html.Div(value, style={**KPI_VALUE_STYLE, "color": value_color}),
        ],
        style={**KPI_STYLE, "borderColor": border},
    )


KPI_STYLE = {
    "backgroundColor": colors.BACKGROUND_CARD,
    "border": f"1px solid {colors.BORDER}",
    "borderRadius": "8px",
    "padding": "10px 14px",
    "minWidth": "0",
}


def section(title, body, width="100%"):
    return html.Div(
        children=[
            html.Div(title, style=SECTION_TITLE_STYLE),
            body,
        ],
        style={**SECTION_STYLE, "width": width, "boxSizing": "border-box"},
    )


def top_metrics_row(account, equity_curve, cycle, circuit, exposure_pct):
    cur_equity = equity_curve[-1]["equity"] if equity_curve else None
    pnl = (cur_equity - 100000.0) if cur_equity is not None else 0.0
    drawdown = equity_curve[-1]["drawdown_pct"] if equity_curve else 0.0
    bp = 0.0
    if account.get("ok") and account.get("buying_power") is not None:
        try:
            bp = float(account["buying_power"])
        except (TypeError, ValueError):
            bp = 0.0
    liq = float((cycle or {}).get("available_liquidity", 0.0) or 0.0)
    liq_ok = liq >= 5000.0
    regime = str((cycle or {}).get("regime", "n/a") or "n/a")
    verdict = str(circuit.get("verdict", "ALLOW") or "ALLOW")
    gate_alert = verdict in ("BLOCK", "FLATTEN", "REDUCE")
    pnl_sign = "+" if pnl >= 0 else ""
    return html.Div(
        children=[
            kpi("EQUITY", "${:,.2f}".format(cur_equity) if cur_equity is not None else "n/a", alert=drawdown <= -0.15),
            kpi("P&L", pnl_sign + "${:,.2f}".format(pnl), alert=pnl < 0),
            kpi("EXPOSURE", "{:.1%}".format(exposure_pct)),
            kpi("LIQUIDITY", "PASS" if liq_ok else "FAIL", alert=not liq_ok),
            kpi("REGIME", regime, alert=regime in ("R04", "R07")),
            kpi("GATE", verdict, alert=gate_alert),
        ],
        style=KPI_GRID_STYLE,
    )


def ai_decision_card(cycle, audit_event):
    payload = (audit_event or {}).get("payload", {}) or {}
    if audit_event is not None:
        symbol = str(audit_event.get("symbol", "n/a") or "n/a")
    else:
        symbol = str((cycle or {}).get("symbol", "n/a") or "n/a")
    ensemble = float((cycle or {}).get("ensemble_signal", payload.get("ensemble_signal", 0.0)) or 0.0)
    confidence = float((cycle or {}).get("effective_confidence", payload.get("confidence", 0.0)) or 0.0)
    uncertainty = 1.0 - confidence
    regime = str((cycle or {}).get("regime", "n/a") or "n/a")
    verdict = str((cycle or {}).get("capital_gate_verdict", payload.get("verdict", "ALLOW")) or "ALLOW")
    action = str((cycle or {}).get("position_action", payload.get("action", "HOLD")) or "HOLD")
    qty = float((cycle or {}).get("quantity", payload.get("quantity", 0.0)) or 0.0)
    deployment = qty * 100000.0
    alert_verdicts = ("BLOCK", "FLATTEN", "REDUCE")
    return html.Div(
        children=[
            html.Div(
                children=[
                    kpi("SYMBOL", symbol),
                    kpi("ENSEMBLE", "{:+.3f}".format(ensemble)),
                    kpi("CONFIDENCE", "{:.2f}".format(confidence)),
                    kpi("UNCERTAINTY", "{:.2f}".format(uncertainty)),
                    kpi("VERDICT", verdict, alert=verdict in alert_verdicts),
                    kpi("DEPLOYMENT", "${:,.0f}".format(deployment)),
                ],
                style=KPI_GRID_STYLE,
            ),
            html.Div(
                children=[
                    html.Span("REGIME", style={**KPI_LABEL_STYLE, "marginRight": "8px"}),
                    html.Span(regime, style={"fontFamily": colors.FONT_MONO, "color": colors.TEXT_PRIMARY, "marginRight": "20px"}),
                    html.Span("ACTION", style={**KPI_LABEL_STYLE, "marginRight": "8px"}),
                    html.Span(action, style={"fontFamily": colors.FONT_MONO, "color": colors.TEXT_PRIMARY}),
                ],
                style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "8px"},
            ),
        ]
    )


def risk_gates_panel(gates, circuit):
    rows = []
    for g in gates:
        status = g.get("status", "PASS")
        bg = colors.SERIES_LONG
        if status == "WARN":
            bg = colors.ALERT_WARN
        elif status == "FAIL":
            bg = colors.ALERT_BADGE
        rows.append(
            html.Div(
                children=[
                    html.Span(g.get("gate_id", "?"), style={"flex": "1", "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
                    html.Span(status, style={"backgroundColor": bg, "color": colors.BACKGROUND_PRIMARY, "padding": "2px 8px", "borderRadius": "4px", "fontWeight": "700", "fontSize": "11px", "marginRight": "10px"}),
                    html.Span(g.get("detail", ""), style={"color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "11px"}),
                ],
                style={"display": "flex", "alignItems": "center", "padding": "6px 0", "borderBottom": "1px solid " + colors.BORDER},
            )
        )
    level = circuit.get("level", "NORMAL")
    level_bg = colors.SERIES_BENCHMARK
    if level == "NORMAL":
        level_bg = colors.SERIES_LONG
    elif level == "WARN":
        level_bg = colors.ALERT_WARN
    elif level == "CRITICAL":
        level_bg = colors.ALERT_BADGE
    elif level == "FLATTEN":
        level_bg = colors.ALERT_CRITICAL
    label = circuit.get("label", "LEVEL 0 - NORMAL")
    header = html.Div(
        children=[
            html.Span("CIRCUIT BREAKER", style={**KPI_LABEL_STYLE, "marginRight": "12px"}),
            html.Span(label, style={"backgroundColor": level_bg, "color": colors.BACKGROUND_PRIMARY, "padding": "3px 10px", "borderRadius": "4px", "fontWeight": "700", "fontSize": "11px"}),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "10px"},
    )
    return html.Div(children=[header, html.Div(children=rows)])


def alpaca_summary_block(account, positions_payload, exposure_pct):
    if not account.get("ok"):
        return html.Div(
            "Alpaca account unavailable: " + str(account.get("error", "unknown")),
            style={"color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO},
        )
    bp = 0.0
    if account.get("buying_power") is not None:
        try:
            bp = float(account["buying_power"])
        except (TypeError, ValueError):
            bp = 0.0
    equity = 0.0
    if account.get("equity") is not None:
        try:
            equity = float(account["equity"])
        except (TypeError, ValueError):
            equity = 0.0
    cash = 0.0
    if account.get("cash") is not None:
        try:
            cash = float(account["cash"])
        except (TypeError, ValueError):
            cash = 0.0
    return html.Div(
        children=[
            kpi("ACCOUNT EQUITY", "${:,.2f}".format(equity)),
            kpi("CASH", "${:,.2f}".format(cash)),
            kpi("BUYING POWER", "${:,.2f}".format(bp)),
            kpi("EXPOSURE", "{:.1%}".format(exposure_pct)),
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "10px"},
    )


def two_col(left_section, right_section):
    return html.Div(
        children=[
            html.Div(left_section, style={"flex": "1", "minWidth": "0"}),
            html.Div(right_section, style={"flex": "1", "minWidth": "0"}),
        ],
        style={"display": "flex", "gap": "12px", "marginBottom": "14px"},
    )


def build_control_room(
    account,
    equity_curve,
    cycle,
    charges,
    agents,
    kalman,
    regime_card,
    circuit,
    gates,
    llm_rows,
    trade_outcome_rows,
    reputation_rows,
    waterfall,
    options_rows,
    positions_payload,
    exposure_pct,
    audit_event,
    equity_fig,
    soc_fig,
    agents_fig,
    kalman_fig,
    regime_fig,
    llm_fig,
    options_fig,
    outcome_fig,
    reputation_fig,
    waterfall_fig,
):
    return html.Div(
        children=[
            section("CURRENT AI DECISION", ai_decision_card(cycle, audit_event)),
            section("PORTFOLIO EQUITY", dcc.Graph(figure=equity_fig, config={"displaylogo": False}, style={"height": "320px"})),
            two_col(
                section("7 SPECIALIST AGENTS (LATEST CYCLE)", dcc.Graph(figure=agents_fig, config={"displaylogo": False}, style={"height": "320px"})),
                section("7-STATE CAPITAL GATE (STATE-OF-CHARGE)", dcc.Graph(figure=soc_fig, config={"displaylogo": False}, style={"height": "320px"})),
            ),
            two_col(
                section("KALMAN ESTIMATION (PRIOR -> POSTERIOR)", dcc.Graph(figure=kalman_fig, config={"displaylogo": False}, style={"height": "320px"})),
                section("MARKET REGIME", dcc.Graph(figure=regime_fig, config={"displaylogo": False}, style={"height": "320px"})),
            ),
            section("RISK CONTROL & CIRCUIT BREAKER", risk_gates_panel(gates, circuit)),
            section("LLM PROVIDERS / FAILOVER / TOKEN USAGE", dcc.Graph(figure=llm_fig, config={"displaylogo": False}, style={"height": "240px"})),
            section("ALPACA PAPER ACCOUNT", alpaca_summary_block(account, positions_payload, exposure_pct)),
            section("ALPACA OPTIONS ACTIVITY", dcc.Graph(figure=options_fig, config={"displaylogo": False}, style={"height": "220px"})),
            two_col(
                section("TRADE OUTCOME LEARNING (LAST 50 CLOSED)", dcc.Graph(figure=outcome_fig, config={"displaylogo": False}, style={"height": "320px"})),
                section("AGENT REPUTATION (BETA-BERNOULLI)", dcc.Graph(figure=reputation_fig, config={"displaylogo": False}, style={"height": "320px"})),
            ),
            section("WHY DID X QUANT X TRADE? (ONE DECISION END-TO-END)", dcc.Graph(figure=waterfall_fig, config={"displaylogo": False}, style={"height": "auto"})),
        ]
    )


def build_stop_session_modal():
    return dcc.ConfirmDialog(
        id="stop-session-modal",
        message=(
            "This dashboard is monitoring-only and does not own the running "
            "trading loop. To stop trading, stop the run_agent.py process."
        ),
    )
