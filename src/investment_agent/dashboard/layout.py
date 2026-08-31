"""X Quant X dashboard layout.

WHAT
====
Builds the persistent header, sidebar navigation, and the six panel layouts
(P01-P06) defined in
alpaca_paper_trading_specifications_x_quant_x/004_xquantx_ui_navigation.txt,
styled per 002_xquantx_aesthetics.txt (card surfaces, Inter/JetBrains Mono
typography, bg-secondary sidebar, border tokens).

WHY
===
Monitoring-only by construction: nothing in this module can submit an order,
modify a position, or change risk rules (004 doc Section 1.4.2). The one
"Stop Session" control is informational only -- the dashboard does not own
the running trading-loop process (that's run_agent.py), so it explains how
to stop it rather than pretending to control it.

HOW
===
Plain dash.html / dash.dcc components only (no extra UI framework
dependency). Panel content builders take pre-computed data (from
data_loader.py) and pre-built Plotly figures (from charts.py) -- this module
does no data fetching or chart math of its own.
"""

from __future__ import annotations

from typing import Optional

from dash import dcc, html

from . import colors

PANELS = [
    ("P01", "Portfolio Overview"),
    ("P02", "Regime Monitor"),
    ("P03", "Signal Dashboard"),
    ("P04", "Position Detail"),
    ("P05", "Order History"),
    ("P06", "Risk Audit"),
]

SIDEBAR_STYLE = {
    "width": "230px", "minHeight": "100vh", "backgroundColor": colors.BACKGROUND_SECONDARY,
    "borderRight": f"1px solid {colors.BORDER}", "padding": "20px 0", "flexShrink": "0",
}
NAV_ITEM_STYLE = {
    "padding": "11px 22px", "cursor": "pointer", "color": colors.TEXT_SECONDARY,
    "fontFamily": colors.FONT_SANS, "fontSize": "13.5px", "fontWeight": "500",
    "borderLeft": "3px solid transparent", "transition": "background-color 150ms",
}
NAV_ITEM_ACTIVE_STYLE = {
    **NAV_ITEM_STYLE,
    "backgroundColor": colors.BACKGROUND_CARD, "color": colors.SERIES_EQUITY,
    "borderLeft": f"3px solid {colors.SERIES_EQUITY}", "fontWeight": "700",
}

HEADER_STYLE = {
    "display": "flex", "alignItems": "center", "justifyContent": "space-between",
    "padding": "14px 28px", "backgroundColor": colors.BACKGROUND_SECONDARY,
    "borderBottom": f"1px solid {colors.BORDER}", "color": colors.TEXT_PRIMARY,
    "fontFamily": colors.FONT_SANS,
}

PAGE_STYLE = {
    "backgroundColor": colors.BACKGROUND_PRIMARY, "minHeight": "100vh",
    "color": colors.TEXT_PRIMARY, "fontFamily": colors.FONT_SANS,
}

CARD = {
    "backgroundColor": colors.BACKGROUND_CARD, "border": f"1px solid {colors.BORDER}",
    "borderRadius": "10px", "overflow": "hidden",
}


def build_header(session_id: str, mode: str = "PAPER") -> html.Div:
    mode_bg = {"PAPER": colors.SERIES_BENCHMARK, "BACKTEST": colors.SERIES_EQUITY, "LIVE": colors.ALERT_CRITICAL}.get(mode, colors.SERIES_BENCHMARK)
    mode_fg = colors.BACKGROUND_PRIMARY if mode == "LIVE" else colors.TEXT_PRIMARY
    return html.Div([
        html.Div([
            html.Span("X QUANT X", style={"fontWeight": "800", "letterSpacing": "2px", "marginRight": "20px", "fontSize": "15px"}),
            html.Span(f"session: {session_id}", style={"marginRight": "18px", "color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
            html.Span(mode, style={"backgroundColor": mode_bg, "color": mode_fg, "padding": "3px 12px", "borderRadius": "5px", "marginRight": "18px", "fontWeight": "700", "fontSize": "11px", "letterSpacing": "1px"}),
            html.Span(f"ruleset: {colors.RULESET_VERSION}", style={"color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span(id="dashboard-elapsed", style={"marginRight": "18px", "color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
            html.Span(id="dashboard-last-refresh", style={"color": colors.TEXT_SECONDARY, "fontFamily": colors.FONT_MONO, "fontSize": "12px"}),
        ]),
    ], style=HEADER_STYLE)


def build_sidebar(active_panel: str) -> html.Div:
    items = []
    for i, (panel_id, panel_name) in enumerate(PANELS, start=1):
        style = NAV_ITEM_ACTIVE_STYLE if panel_id == active_panel else NAV_ITEM_STYLE
        items.append(html.Div(f"{i}  ·  {panel_name}", id={"type": "nav-item", "panel": panel_id}, style=style, n_clicks=0))
    items.append(html.Div(
        "STOP SESSION", id="stop-session-btn", n_clicks=0,
        style={
            **NAV_ITEM_STYLE, "marginTop": "44px", "color": colors.ALERT_WARN,
            "borderTop": f"1px solid {colors.BORDER}", "paddingTop": "18px",
            "fontSize": "12px", "letterSpacing": "1px", "fontWeight": "700",
        },
    ))
    return html.Div(items, style=SIDEBAR_STYLE)


def build_metric_card(label: str, value: str, alert: bool = False) -> html.Div:
    return html.Div([
        html.Div(label.upper(), style={"fontSize": "11px", "color": colors.TEXT_SECONDARY, "letterSpacing": "0.5px", "fontWeight": "600"}),
        html.Div(value, style={
            "fontSize": "22px", "fontFamily": colors.FONT_MONO,
            "color": colors.ALERT_CRITICAL if alert else colors.TEXT_PRIMARY, "fontWeight": "700", "marginTop": "4px",
        }),
    ], style={
        **CARD, "padding": "14px 18px", "flex": "1",
        "borderColor": colors.ALERT_CRITICAL if alert else colors.BORDER,
    })


def _chart_card(fig, height: Optional[str] = None) -> html.Div:
    style = {**CARD, "flex": "1"}
    graph_style = {"height": height} if height else {}
    return html.Div(dcc.Graph(figure=fig, style=graph_style, config={"displaylogo": False}), style=style)


def panel_p01(account: dict, equity_curve: list, treemap_fig, equity_fig, drawdown_fig) -> html.Div:
    current_equity = equity_curve[-1]["equity"] if equity_curve else None
    drawdown_pct = equity_curve[-1]["drawdown_pct"] if equity_curve else 0.0
    alert = drawdown_pct <= -0.15
    equity_text = f"${current_equity:,.2f}" if current_equity is not None else "n/a"
    bp_text = f"${float(account['buying_power']):,.2f}" if account.get("ok") and account.get("buying_power") is not None else "n/a"

    return html.Div([
        html.Div([
            build_metric_card("Equity", equity_text, alert=alert),
            build_metric_card("Drawdown", f"{drawdown_pct:.1%}", alert=alert),
            build_metric_card("Buying Power", bp_text),
            build_metric_card("Account Status", account.get("status", "n/a") if account.get("ok") else "unavailable"),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),
        html.Div([
            _chart_card(equity_fig),
            _chart_card(drawdown_fig),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),
        _chart_card(treemap_fig, height="360px"),
    ])


def panel_p02(regime_fig, history_fig, gauge_fig) -> html.Div:
    return html.Div([
        html.Div(_chart_card(regime_fig), style={"marginBottom": "12px"}),
        html.Div([
            html.Div(_chart_card(history_fig), style={"flex": "2"}),
            html.Div(_chart_card(gauge_fig), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),
    ])


def panel_p03(signal_fig) -> html.Div:
    return _chart_card(signal_fig, height="72vh")


def panel_p04(position_table_fig) -> html.Div:
    return _chart_card(position_table_fig, height="72vh")


def panel_p05(order_table_fig) -> html.Div:
    return _chart_card(order_table_fig, height="72vh")


def panel_p06(risk_table_fig, correlation_fig) -> html.Div:
    return html.Div([
        html.Div(_chart_card(risk_table_fig, height="52vh"), style={"marginBottom": "12px"}),
        _chart_card(correlation_fig),
    ])


def build_stop_session_modal() -> dcc.ConfirmDialog:
    return dcc.ConfirmDialog(
        id="stop-session-modal",
        message=(
            "This dashboard is monitoring-only and does not own the running "
            "trading loop. To actually stop trading, stop the run_agent.py "
            "process directly (Ctrl+C in its terminal)."
        ),
    )

