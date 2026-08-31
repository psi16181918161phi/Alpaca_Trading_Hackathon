"""X Quant X monitoring dashboard -- Dash application entrypoint.

WHAT
====
Wires together data_loader.py, charts.py, and layout.py into a running Dash
app implementing the six-panel dashboard from
alpaca_paper_trading_specifications_x_quant_x/004_xquantx_ui_navigation.txt:
Portfolio Overview, Regime Monitor, Signal Dashboard, Position Detail, Order
History, Risk Audit.

WHY
===
This is the monitoring surface for the whole trading system: it never
imports the Alpaca TradingClient directly (see data_loader.py's docstring)
and contains no trading logic of its own, per 005_xquantx_coding_standards.txt.

HOW
===
- Sidebar + keyboard shortcuts (1-6) switch the active panel via a dcc.Store.
- A dcc.Interval refreshes data at the configured per-bar interval (default
  60s, matching 004 doc Section 1.3's "auto-refresh at the per-bar interval").
- "R" forces an immediate refresh; keyboard listening is registered once via
  a small inline script (Dash 2.17+ / this app's dash>=4's
  `dash_clientside.set_props`), not a polling callback.

Run with:  python run_dashboard.py   (from the repository root)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from dash import Dash, Input, Output, State, ALL, ctx, dcc, html

from . import charts, colors, data_loader, layout

REFRESH_INTERVAL_MS = 60_000  # per-bar interval, matches 004 doc default (Paper mode)
SESSION_ID = f"session-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
MODE = "PAPER"  # only paper trading is implemented; mode-switch UI is out of MVP scope per spec 004 Section 1.9.1
SESSION_START = datetime.now()

app = Dash(__name__, suppress_callback_exceptions=True, title="X Quant X Dashboard")
server = app.server  # exposed for wsgi/gunicorn if ever needed

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href=\"""" + colors.GOOGLE_FONTS_HREF + """\" rel="stylesheet">
        <style>
            body { margin: 0; background-color: """ + colors.BACKGROUND_PRIMARY + """; font-family: """ + colors.FONT_SANS + """; }
            ::-webkit-scrollbar { width: 10px; height: 10px; }
            ::-webkit-scrollbar-track { background: """ + colors.BACKGROUND_PRIMARY + """; }
            ::-webkit-scrollbar-thumb { background: """ + colors.BORDER + """; border-radius: 5px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
        <script>
        document.addEventListener('keydown', function(e) {
            var panelKeys = {'1': 'P01', '2': 'P02', '3': 'P03', '4': 'P04', '5': 'P05', '6': 'P06'};
            if (panelKeys[e.key] && window.dash_clientside) {
                window.dash_clientside.set_props('active-panel-store', {data: panelKeys[e.key]});
            }
            if (e.key === 'r' || e.key === 'R') {
                if (window.dash_clientside) {
                    window.dash_clientside.set_props('refresh-interval', {n_intervals: Date.now()});
                }
            }
        });
        </script>
    </body>
</html>"""


def _build_panel_content(panel_id: str) -> html.Div:
    history = data_loader.load_trade_history()
    equity_curve = data_loader.compute_equity_curve(history)
    account = data_loader.get_account_summary_safe()
    positions = data_loader.get_positions_safe().get("positions", [])
    orders = data_loader.get_order_history_safe().get("orders", [])
    risk_log = data_loader.get_risk_gate_log(history)

    if panel_id == "P01":
        return layout.panel_p01(
            account, equity_curve,
            treemap_fig=charts.build_portfolio_treemap(positions, SESSION_ID, MODE),
            equity_fig=charts.build_equity_curve_chart(equity_curve, SESSION_ID, MODE),
            drawdown_fig=charts.build_drawdown_waterfall_chart(equity_curve, SESSION_ID, MODE),
        )
    if panel_id == "P02":
        return layout.panel_p02(
            regime_fig=charts.build_regime_probability_chart(history, SESSION_ID, MODE),
            history_fig=charts.build_regime_history_timeline(history, session_id=SESSION_ID, mode=MODE),
            gauge_fig=charts.build_entropy_gauge(history, SESSION_ID, MODE),
        )
    if panel_id == "P03":
        return layout.panel_p03(charts.build_signal_score_chart(history, SESSION_ID, MODE))
    if panel_id == "P04":
        return layout.panel_p04(charts.build_position_table(positions, SESSION_ID, MODE))
    if panel_id == "P05":
        return layout.panel_p05(charts.build_order_history_table(orders, SESSION_ID, MODE))
    if panel_id == "P06":
        return layout.panel_p06(
            risk_table_fig=charts.build_risk_gate_table(risk_log, SESSION_ID, MODE),
            correlation_fig=charts.build_correlation_heatmap_chart(history, SESSION_ID, MODE),
        )
    return html.Div(f"Unknown panel: {panel_id}")


app.layout = html.Div([
    dcc.Store(id="active-panel-store", data="P01"),
    dcc.Interval(id="refresh-interval", interval=REFRESH_INTERVAL_MS, n_intervals=0),
    layout.build_header(SESSION_ID, MODE),
    html.Div([
        html.Div(id="sidebar-container"),
        html.Div(id="panel-content", style={"flex": "1", "padding": "20px", "minWidth": "0"}),
    ], style={"display": "flex"}),
    layout.build_stop_session_modal(),
], style=layout.PAGE_STYLE)


@app.callback(
    Output("sidebar-container", "children"),
    Input("active-panel-store", "data"),
)
def _render_sidebar(active_panel):
    return layout.build_sidebar(active_panel)


@app.callback(
    Output("active-panel-store", "data"),
    Input({"type": "nav-item", "panel": ALL}, "n_clicks"),
    State("active-panel-store", "data"),
    prevent_initial_call=True,
)
def _handle_nav_click(n_clicks_list, current_panel):
    triggered = ctx.triggered_id
    if not triggered or not any(n_clicks_list):
        return current_panel
    return triggered["panel"]


@app.callback(
    Output("panel-content", "children"),
    Input("active-panel-store", "data"),
    Input("refresh-interval", "n_intervals"),
)
def _render_panel(active_panel, _n_intervals):
    return _build_panel_content(active_panel)


@app.callback(
    Output("dashboard-elapsed", "children"),
    Output("dashboard-last-refresh", "children"),
    Input("refresh-interval", "n_intervals"),
)
def _update_header_clock(_n_intervals):
    elapsed = datetime.now() - SESSION_START
    elapsed_str = str(elapsed).split(".")[0]
    last_refresh = datetime.now().strftime("%H:%M:%S")
    return f"elapsed: {elapsed_str}", f"last refresh: {last_refresh}"


@app.callback(
    Output("stop-session-modal", "displayed"),
    Input("stop-session-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _show_stop_modal(n_clicks):
    return bool(n_clicks)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
