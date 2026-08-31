"""X Quant X dashboard layout.

WHAT
====
Builds the persistent header, sidebar navigation, and the six panel layouts
(P01-P06) defined in
alpaca_paper_trading_specifications_x_quant_x/004_xquantx_ui_navigation.txt.

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
    "width": "220px", "minHeight": "100vh", "backgroundColor": colors.BACKGROUND_PRIMARY,
    "borderRight": f"1px solid {colors.GRID_LINE}", "padding": "16px 0", "flexShrink": "0",
}
NAV_ITEM_STYLE = {
    "padding": "10px 20px", "cursor": "pointer", "color": colors.TEXT_PRIMARY,
    "fontFamily": "monospace", "fontSize": "14px",
}
NAV_ITEM_ACTIVE_STYLE = {**NAV_ITEM_STYLE, "backgroundColor": colors.SERIES_EQUITY, "color": colors.BACKGROUND_PRIMARY, "fontWeight": "bold"}

HEADER_STYLE = {
    "display": "flex", "alignItems": "center", "justifyContent": "space-between",
    "padding": "10px 24px", "backgroundColor": colors.BACKGROUND_PRIMARY,
    "borderBottom": f"1px solid {colors.GRID_LINE}", "color": colors.TEXT_PRIMARY,
    "fontFamily": "monospace",
}

PAGE_STYLE = {"backgroundColor": colors.BACKGROUND_PRIMARY, "minHeight": "100vh", "color": colors.TEXT_PRIMARY, "fontFamily": "sans-serif"}


def build_header(session_id: str, mode: str = "PAPER") -> html.Div:
    mode_bg = {"PAPER": colors.SERIES_BENCHMARK, "BACKTEST": colors.SERIES_EQUITY, "LIVE": colors.ALERT_CRITICAL}.get(mode, colors.SERIES_BENCHMARK)
    return html.Div([
        html.Div([
            html.Span("X QUANT X", style={"fontWeight": "bold", "letterSpacing": "2px", "marginRight": "16px"}),
            html.Span(f"session: {session_id}", style={"marginRight": "16px", "color": colors.GRID_LINE}),
            html.Span(mode, style={"backgroundColor": mode_bg, "padding": "2px 10px", "borderRadius": "4px", "marginRight": "16px"}),
            html.Span(f"ruleset: {colors.RULESET_VERSION}", style={"color": colors.GRID_LINE}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span(id="dashboard-elapsed", style={"marginRight": "16px"}),
            html.Span(id="dashboard-last-refresh"),
        ]),
    ], style=HEADER_STYLE)


def build_sidebar(active_panel: str) -> html.Div:
    items = []
    for i, (panel_id, panel_name) in enumerate(PANELS, start=1):
        style = NAV_ITEM_ACTIVE_STYLE if panel_id == active_panel else NAV_ITEM_STYLE
        items.append(html.Div(f"{i}. {panel_name}", id={"type": "nav-item", "panel": panel_id}, style=style, n_clicks=0))
    items.append(html.Div(
        "Stop Session", id="stop-session-btn", n_clicks=0,
        style={**NAV_ITEM_STYLE, "marginTop": "40px", "color": colors.ALERT_WARN, "borderTop": f"1px solid {colors.GRID_LINE}", "paddingTop": "16px"},
    ))
    return html.Div(items, style=SIDEBAR_STYLE)


def build_metric_header(label: str, value: str, alert: bool = False) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": colors.GRID_LINE}),
        html.Div(value, style={"fontSize": "20px", "color": colors.ALERT_CRITICAL if alert else colors.TEXT_PRIMARY, "fontWeight": "bold"}),
    ], style={"padding": "8px 16px"})


def panel_p01(account: dict, equity_curve: list, treemap_fig, equity_fig, drawdown_fig) -> html.Div:
    current_equity = equity_curve[-1]["equity"] if equity_curve else None
    drawdown_pct = equity_curve[-1]["drawdown_pct"] if equity_curve else 0.0
    alert = drawdown_pct <= -0.15
    equity_text = f"${current_equity:,.2f}" if current_equity is not None else "n/a"
    bp_text = f"${float(account['buying_power']):,.2f}" if account.get("ok") and account.get("buying_power") is not None else "n/a"

    header_bg = colors.ALERT_CRITICAL if alert else "transparent"
    return html.Div([
        html.Div([
            build_metric_header("Equity", equity_text, alert=alert),
            build_metric_header("Drawdown", f"{drawdown_pct:.1%}", alert=alert),
            build_metric_header("Buying Power", bp_text),
            build_metric_header("Account Status", account.get("status", "n/a") if account.get("ok") else "unavailable"),
        ], style={"display": "flex", "backgroundColor": header_bg, "borderRadius": "6px"}),
        html.Div([
            html.Div(dcc.Graph(figure=equity_fig), style={"flex": "1"}),
            html.Div(dcc.Graph(figure=drawdown_fig), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "8px"}),
        dcc.Graph(figure=treemap_fig),
    ])


def panel_p02(regime_fig, history_fig, gauge_fig) -> html.Div:
    return html.Div([
        dcc.Graph(figure=regime_fig),
        html.Div([
            html.Div(dcc.Graph(figure=history_fig), style={"flex": "2"}),
            html.Div(dcc.Graph(figure=gauge_fig), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "8px"}),
    ])


def panel_p03(signal_fig) -> html.Div:
    return html.Div([dcc.Graph(figure=signal_fig, style={"height": "70vh"})])


def panel_p04(position_table_fig) -> html.Div:
    return html.Div([dcc.Graph(figure=position_table_fig, style={"height": "70vh"})])


def panel_p05(order_table_fig) -> html.Div:
    return html.Div([dcc.Graph(figure=order_table_fig, style={"height": "70vh"})])


def panel_p06(risk_table_fig, correlation_fig) -> html.Div:
    return html.Div([
        dcc.Graph(figure=risk_table_fig, style={"height": "55vh"}),
        dcc.Graph(figure=correlation_fig),
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
