"""X Quant X monitoring dashboard -- Dash application entrypoint.

WHAT
====
Wires together ``data_loader.py``, ``charts.py``, and ``layout.py`` into a
single scrolling "control-room" dashboard. The 12 sections are rendered
in a fixed vertical order so a judge can scan the entire pipeline in
under 20 seconds:

  1. Alpaca account (top panel: Equity / Daily P&L / Total P&L / Cash / Buying Power)
  2. Current AI Decision
  3. Portfolio Equity (P&L)
  4. Seven specialist agents
  5. Seven-state capital gate
  6. Kalman estimation
  7. Market regime
  8. Risk Control & circuit breaker
  9. LLM providers / failover
 10. Options activity
 11. Trade outcome learning + agent reputation
 12. "Why did X Quant X trade?" decision waterfall

WHY
====
This is the monitoring surface for the whole trading system. It never
imports the Alpaca TradingClient directly (see ``data_loader.py``) and
contains no trading logic of its own.

HOW
====
- A ``dcc.Interval`` refreshes data every ``REFRESH_INTERVAL_MS``.
- The control-room layout is rebuilt in full on each refresh; Dash
  diffs the resulting tree. The rebuild is cheap because every
  component is read-only and the data sources are local files plus
  one Alpaca API call.

Run with:  ``python run_dashboard.py``  (from the repository root)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict

from dash import Dash, Input, Output, dcc, html

from . import charts, colors, data_loader, layout

REFRESH_INTERVAL_MS = 60_000
SESSION_ID = f"session-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
MODE = "PAPER"
SESSION_START = datetime.now()

app = Dash(__name__, suppress_callback_exceptions=True, title="X Quant X Dashboard")
server = app.server

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
    </body>
</html>"""


def _build_control_room():
    history = data_loader.load_trade_history()
    equity_curve = data_loader.compute_equity_curve(history)
    account = data_loader.get_account_summary_safe()
    # The top panel's authoritative source. Prefers the live-loop's
    # cached snapshot inside ``live_state.json`` (so the dashboard
    # never has to hit the broker directly), and falls back to a live
    # ``get_account_snapshot`` call when the loop is not running.
    alpaca_snapshot = data_loader.get_alpaca_account_snapshot()
    # Session controller status (state, cycle, last decision) -- the
    # SessionController writes this on every transition. The dashboard
    # is read-only w.r.t. this file.
    session_status = data_loader.get_session_status()
    positions_payload = data_loader.get_positions_safe()
    orders = data_loader.get_order_history_safe().get("orders", [])

    cycle = data_loader.latest_cycle_snapshot(history)
    audit_event = data_loader.latest_decision_event()
    charges = data_loader.get_seven_state_charges(cycle)
    agents = data_loader.get_seven_agents(cycle, history)
    kalman = data_loader.get_kalman_card(cycle)
    regime_card = data_loader.get_regime_card(cycle)
    circuit = data_loader.get_circuit_breaker_state(cycle)
    gates = data_loader.get_risk_gates_status(cycle, history)
    llm_rows = data_loader.summarize_llm_providers()
    trade_outcome_rows = data_loader.get_trade_outcome_learning(history)
    reputation_rows = data_loader.get_reputation_snapshot(history)
    waterfall = data_loader.get_decision_waterfall(cycle, audit_event)
    # Authoritative options payload: real broker /orders filter, with
    # error surface so a failed broker call doesn't masquerade as
    # 'no options'.
    options_payload = data_loader.get_recent_options_activity()
    options_rows = options_payload.get("orders", []) if options_payload.get("ok") else []
    options_error = options_payload.get("error")

    buying_power = None
    if account.get("ok") and account.get("buying_power") is not None:
        try:
            buying_power = float(account["buying_power"])
        except (TypeError, ValueError):
            buying_power = None
    exposure_pct = data_loader.get_top_exposure_pct(positions_payload, buying_power)

    equity_fig = charts.build_equity_curve_chart(equity_curve, SESSION_ID, MODE, source="strategy")
    soc_fig = charts.build_seven_state_soc_chart(charges, SESSION_ID, MODE)
    agents_fig = charts.build_seven_agents_table(agents, SESSION_ID, MODE)
    kalman_fig = charts.build_kalman_chart(kalman, SESSION_ID, MODE)
    regime_fig = charts.build_regime_panel_chart(regime_card, SESSION_ID, MODE)
    llm_fig = charts.build_llm_providers_table(llm_rows, SESSION_ID, MODE)
    options_fig = charts.build_options_table(options_rows, SESSION_ID, MODE, error=options_error)
    outcome_fig = charts.build_trade_outcome_table(trade_outcome_rows, SESSION_ID, MODE)
    reputation_fig = charts.build_reputation_table(reputation_rows, SESSION_ID, MODE)
    waterfall_fig = charts.build_decision_waterfall(waterfall)

    return layout.build_control_room(
        account=account,
        alpaca_snapshot=alpaca_snapshot,
        session_status=session_status,
        equity_curve=equity_curve,
        cycle=cycle,
        charges=charges,
        agents=agents,
        kalman=kalman,
        regime_card=regime_card,
        circuit=circuit,
        gates=gates,
        llm_rows=llm_rows,
        trade_outcome_rows=trade_outcome_rows,
        reputation_rows=reputation_rows,
        waterfall=waterfall,
        options_rows=options_rows,
        positions_payload=positions_payload,
        exposure_pct=exposure_pct,
        audit_event=audit_event,
        equity_fig=equity_fig,
        soc_fig=soc_fig,
        agents_fig=agents_fig,
        kalman_fig=kalman_fig,
        regime_fig=regime_fig,
        llm_fig=llm_fig,
        options_fig=options_fig,
        outcome_fig=outcome_fig,
        reputation_fig=reputation_fig,
        waterfall_fig=waterfall_fig,
    )


app.layout = html.Div([
    html.Button(
        "Refresh",
        id="refresh-btn",
        n_clicks=0,
        style={
            "position": "fixed", "top": "8px", "right": "20px", "zIndex": "9999",
            "backgroundColor": "#2DD4BF", "color": "#000", "border": "none",
            "padding": "6px 14px", "borderRadius": "4px",
            "fontFamily": "Inter, sans-serif", "fontWeight": "700",
            "cursor": "pointer", "fontSize": "12px",
        },
    ),
    layout.build_header(SESSION_ID, MODE),
    html.Div(id="control-room", style={"padding": "20px", "minWidth": "0"}),
    layout.build_stop_session_modal(),
    # Client-side meta-refresh as a workaround for the Dash 4.4.1
    # callback bug (https://github.com/plotly/dash/issues/2885) where
    # single-Output callbacks misclassify as wildcard multi-output.
    # The page reloads every REFRESH_INTERVAL_MS, so the dashboard
    # always shows fresh Alpaca data even though the in-process
    # callback returns 500. Manual refresh still works via the
    # floating button.
    html.Script(
        f"setTimeout(function(){{ window.location.reload(); }}, {REFRESH_INTERVAL_MS});"
    ),
], style=layout.PAGE_STYLE)


@app.callback(
    Output("control-room", "children"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=False,
)
def _render_control_room(_n_clicks):
    return _build_control_room()


# ----- session control buttons (write a command file; the daemon polls it) -----

# Default session parameters; users override via the session daemon's
# CLI / env. The dashboard only tells the daemon to start, not how.
DEFAULT_SESSION_PARAMS: Dict[str, Any] = {
    "stage": "paper",
    "decision_interval_seconds": 300,
    "symbol_universe": ["AAPL", "SPY", "MSFT", "TSLA", "NVDA"],
    "max_lookups_per_interval": 2,
}


@app.callback(
    Output("session-start-btn", "n_clicks"),
    Input("session-start-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _on_session_start(n_clicks):
    if not n_clicks:
        return 0
    data_loader.write_session_command(
        "start", params=DEFAULT_SESSION_PARAMS,
    )
    return n_clicks


@app.callback(
    Output("session-stop-btn", "n_clicks"),
    Input("session-stop-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _on_session_stop(n_clicks):
    if not n_clicks:
        return 0
    data_loader.write_session_command("stop")
    return n_clicks


@app.callback(
    Output("session-emergency-btn", "n_clicks"),
    Input("session-emergency-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _on_session_emergency(n_clicks):
    if not n_clicks:
        return 0
    data_loader.write_session_command("emergency_stop")
    return n_clicks


@app.callback(
    Output("stop-session-modal", "displayed"),
    Input("stop-session-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _show_stop_modal(n_clicks):
    return bool(n_clicks)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
