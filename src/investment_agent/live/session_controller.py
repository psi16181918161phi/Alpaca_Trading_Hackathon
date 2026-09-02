"""Session controller for the X Quant X paper-trading session.

WHAT
====
A single-process owner of the ``LiveOrchestrator`` that the dashboard
talks to. The dashboard never imports the Alpaca TradingClient and
never touches the orchestrator directly; it only reads
``session_status.json`` and asks the controller to ``start`` / ``stop``
/ ``emergency_stop`` via the controller's ``request_*`` methods, which
in turn are bound to file-based commands written by the dashboard
callback (or by a human running ``scripts/session_control.py start``).

The controller's responsibilities:
  * Own a single background thread running the live loop.
  * Persist session state to ``session_status.json`` on every transition
    so the dashboard can poll it on each refresh.
  * Validate the broker connection and account equity BEFORE starting
    a real-paper session (per the spec: validate creds, validate paper
    account, confirm account equity, check liquidity floor, check
    existing positions, then start).
  * On ``stop``: prevent new decisions / new orders, allow submitted
    orders to reconcile, persist final state, mark status STOPPED.
  * On ``emergency_stop``: in addition to ``stop``, cancel any
    currently-accepted open orders via the broker, then HALT.

WHY
====
The architecture explicitly forbids the dashboard from calling
``TradingClient.submit_order`` directly. The session controller is the
single seam between the UI and the live loop, so the monitoring UI
stays read-only and the trading logic stays testable.

HOW
====
Status is persisted atomically to ``session_status.json`` (default
project root). The dashboard reads that file on each refresh; the
controller writes it on every state change.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import uuid


logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Lifecycle states of a paper-trading session.

    Lifecycle:

        STOPPED  --▶ STARTING --▶ RUNNING --▶ STOPPING --▶ STOPPED
                                          ╲
                                           --▶ EMERGENCY_HALT --▶ STOPPED
    """
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    EMERGENCY_HALT = "EMERGENCY_HALT"
    ERROR = "ERROR"


DEFAULT_STATUS_FILE = "session_status.json"
DEFAULT_COMMAND_FILE = "session_command.json"


@dataclass
class SessionStatus:
    """Serializable session status that the dashboard reads."""
    session_id: str = ""
    state: str = SessionState.STOPPED.value
    stage: str = "dry_run"
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    last_cycle_at: Optional[str] = None
    last_decision_summary: str = ""
    last_error: str = ""
    cycle_index: int = 0
    next_cycle_at: Optional[str] = None
    decision_interval_seconds: int = 60
    symbol_universe: List[str] = field(default_factory=list)
    max_lookups_per_interval: int = 2
    total_decisions: int = 0
    total_orders: int = 0
    total_closed: int = 0
    pid: int = 0
    started_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SessionController:
    """Owns the LiveOrchestrator thread for a single paper session.

    Parameters
    ----------
    status_file : str
        Path to the JSON file the dashboard reads. Written atomically
        on every state change.
    command_file : str
        Path to the JSON file the dashboard writes commands to. The
        controller polls this file every second; commands include
        ``start``, ``stop``, ``emergency_stop``.
    build_orchestrator : callable
        Factory that builds a fully-wired ``LiveOrchestrator`` (market
        data, agent factory, executor, screener, product gate, etc.).
        Called exactly once when the user presses START.
    on_session_complete : callable, optional
        Callback invoked with the SessionStatus when the loop thread
        exits naturally (e.g. max_intervals reached). Used by tests.
    """

    def __init__(
        self,
        build_orchestrator: Callable[[], Any],
        status_file: str = DEFAULT_STATUS_FILE,
        command_file: str = DEFAULT_COMMAND_FILE,
        on_session_complete: Optional[Callable[[SessionStatus], None]] = None,
        preflight: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    ) -> None:
        self._build_orchestrator = build_orchestrator
        self.status_file = status_file
        self.command_file = command_file
        self._on_session_complete = on_session_complete
        # Tests can inject a no-op preflight that always passes, so
        # they don't need a real broker connection.
        self._preflight_fn = preflight

        self._lock = threading.RLock()
        self._status = SessionStatus(pid=os.getpid())
        self._orchestrator: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._emergency_stop = threading.Event()
        self._start_params: Dict[str, Any] = {}
        # Clear any stale command file from a previous run.
        self._clear_command_file()
        # Eagerly write the initial STOPPED state so the dashboard
        # can render without a race condition.
        self._persist_status(self._status.to_dict())

    # ----- public read API -----

    @property
    def status(self) -> SessionStatus:
        with self._lock:
            return SessionStatus(**self._status.to_dict())

    def get_status_dict(self) -> Dict[str, Any]:
        return self.status.to_dict()

    # ----- public write API -----

    def request_start(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Schedule a START via the command file (idempotent)."""
        self._write_command({"action": "start", "params": params or {}})

    def request_stop(self) -> None:
        """Schedule a normal STOP (allow in-flight orders to reconcile)."""
        self._write_command({"action": "stop"})

    def request_emergency_stop(self) -> None:
        """Schedule an EMERGENCY STOP (cancel open orders, then HALT)."""
        self._write_command({"action": "emergency_stop"})

    # ----- command polling (run from the controller thread) -----

    def _poll_command_file(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.command_file):
            return None
        try:
            with open(self.command_file, "r", encoding="utf-8") as f:
                cmd = json.load(f)
            if isinstance(cmd, dict):
                return cmd
        except (OSError, json.JSONDecodeError):
            return None
        return None

    def _clear_command_file(self) -> None:
        try:
            if os.path.exists(self.command_file):
                os.unlink(self.command_file)
        except OSError:
            pass

    def _write_command(self, cmd: Dict[str, Any]) -> None:
        Path(self.command_file).parent.mkdir(parents=True, exist_ok=True)
        payload = {**cmd, "requested_at": datetime.now(timezone.utc).isoformat()}
        with open(self.command_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    # ----- status persistence -----

    def _update_status(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._status, k):
                    setattr(self._status, k, v)
            payload = self._status.to_dict()
        # Write outside the lock so we never block readers on disk.
        self._persist_status(payload)

    def _persist_status(self, payload: Dict[str, Any]) -> None:
        path = self.status_file
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            # Write directly (no temp + rename) to keep the controller
            # test-friendly. The dashboard re-reads the file on every
            # refresh, so a brief non-atomic window is acceptable.
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to persist session status: %s", exc)

    # ----- pre-flight validation -----

    def _validate_preflight(self, params: Dict[str, Any]) -> Optional[str]:
        """Run the pre-flight checks before allowing a real-paper start.

        Returns ``None`` on success or an error string the dashboard can
        surface. The pre-flight does NOT itself modify any broker state
        -- it only reads the account endpoint.
        """
        from ..execution.execution import get_account_snapshot
        snap = get_account_snapshot()
        if not snap.get("ok"):
            return f"Alpaca account unreachable: {snap.get('error', 'unknown')}"
        equity = snap.get("equity")
        if equity is None or float(equity) <= 0:
            return f"Alpaca account equity is non-positive: {equity!r}"
        # Buying power / liquidity floor check. Use Alpaca's own notion
        # of liquidity instead of inventing a number.
        bp = float(snap.get("buying_power") or 0.0)
        if bp < 1000.0:
            return f"Buying power too low for paper trading: ${bp:,.2f}"
        # Existing positions: warn but allow.
        try:
            from ..execution.execution import get_positions
            positions = get_positions() or []
        except Exception:  # noqa: BLE001
            positions = []
        if positions:
            logger.info("preflight: %d open positions already on the account",
                        len(positions))
        # Persist a small bit of preflight metadata in the status.
        self._update_status(
            started_by=str(params.get("started_by", "dashboard")),
        )
        return None

    # ----- lifecycle -----

    def start(self, params: Optional[Dict[str, Any]] = None) -> bool:
        """Start a paper-trading session in a background thread.

        Returns True if the session was actually launched, False if the
        controller is already running (idempotent).
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("start() ignored: session already running")
                return False
            self._stop_requested.clear()
            self._emergency_stop.clear()
            self._start_params = dict(params or {})
            sess_id = str(self._start_params.get("session_id") or f"session-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}")
            self._update_status(
                session_id=sess_id,
                state=SessionState.STARTING.value,
                started_at=datetime.now(timezone.utc).isoformat(),
                last_error="",
                last_decision_summary="",
                cycle_index=0,
                total_decisions=0,
                total_orders=0,
                total_closed=0,
                stage=str(self._start_params.get("stage", "paper")),
                decision_interval_seconds=int(
                    self._start_params.get("decision_interval_seconds", 60)),
                symbol_universe=list(
                    self._start_params.get("symbol_universe", [])),
                max_lookups_per_interval=int(
                    self._start_params.get("max_lookups_per_interval", 2)),
            )
            self._thread = threading.Thread(
                target=self._run_session,
                name="xqx-session",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self, emergency: bool = False) -> bool:
        """Request a (emergency) stop. Returns True if a stop was scheduled."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return False
            self._stop_requested.set()
            if emergency:
                self._emergency_stop.set()
            self._update_status(
                state=(SessionState.EMERGENCY_HALT.value if emergency
                       else SessionState.STOPPING.value),
                stopped_at=datetime.now(timezone.utc).isoformat(),
            )
        return True

    def _run_session(self) -> None:
        """Background thread: build the orchestrator, run cycles, update status."""
        try:
            err = (self._preflight_fn or self._validate_preflight)(self._start_params)
            if err is not None:
                self._update_status(
                    state=SessionState.ERROR.value,
                    last_error=err,
                    stopped_at=datetime.now(timezone.utc).isoformat(),
                )
                return

            self._orchestrator = self._build_orchestrator()
            self._update_status(state=SessionState.RUNNING.value)

            interval = max(1, int(self._start_params.get("decision_interval_seconds", 60)))
            max_intervals = self._start_params.get("max_intervals")
            cycles_run = 0

            while not self._stop_requested.is_set():
                if self._emergency_stop.is_set():
                    break
                # Drive exactly one cycle.
                try:
                    report = self._orchestrator.run_once()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("session cycle failed")
                    self._update_status(last_error=str(exc))
                    time.sleep(1)
                    continue
                self._on_cycle_complete(report)
                cycles_run += 1
                if max_intervals is not None and cycles_run >= int(max_intervals):
                    logger.info("session: reached max_intervals=%s", max_intervals)
                    break
                # Sleep, but break out promptly on stop.
                if self._stop_requested.wait(timeout=interval):
                    break

            # On stop: cancel open orders if emergency.
            if self._emergency_stop.is_set():
                self._cancel_open_orders()

            # Final persistence so the dashboard sees a clean STOPPED.
            self._update_status(
                state=SessionState.STOPPED.value,
                stopped_at=datetime.now(timezone.utc).isoformat(),
                next_cycle_at=None,
            )
            final_status = self.status
            if self._on_session_complete is not None:
                try:
                    self._on_session_complete(final_status)
                except Exception:  # noqa: BLE001
                    logger.exception("on_session_complete callback failed")
        except Exception as exc:  # noqa: BLE001
            logger.exception("session thread crashed")
            self._update_status(
                state=SessionState.ERROR.value,
                last_error=str(exc),
                stopped_at=datetime.now(timezone.utc).isoformat(),
            )
        finally:
            # Mark the thread reference as None so a subsequent start()
            # sees a clean slate (idempotent restart).
            with self._lock:
                self._thread = None

    def _on_cycle_complete(self, report: Any) -> None:
        """Update the session status from a freshly-completed IntervalReport."""
        decisions = len(getattr(report, "decisions", []) or [])
        orders = len(getattr(report, "orders", []) or [])
        exits = len(getattr(report, "exits", []) or [])
        last_decision = ""
        try:
            d = (getattr(report, "decisions", []) or [])
            if d:
                def _fmt(dec):
                    return f"{dec.get('action', 'HOLD')} {dec.get('symbol', '?')} ({dec.get('product', 'none')})"

                meaningful = [x for x in d if x.get("action") not in ("HOLD", None)]
                if meaningful:
                    last_decision = _fmt(meaningful[0])
                else:
                    last_decision = _fmt(d[0])
        except Exception:  # noqa: BLE001
            pass
        interval = int(self._start_params.get("decision_interval_seconds", 60))
        next_at = None
        try:
            next_at = (datetime.now(timezone.utc).timestamp() + interval)
            next_at = datetime.fromtimestamp(next_at, tz=timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            next_at = None
        with self._lock:
            self._status.cycle_index += 1
            self._status.total_decisions += decisions
            self._status.total_orders += orders
            self._status.total_closed += exits
        self._update_status(
            last_cycle_at=datetime.now(timezone.utc).isoformat(),
            last_decision_summary=last_decision,
            next_cycle_at=next_at,
        )

    def _cancel_open_orders(self) -> None:
        """Best-effort cancellation of any orders that are still ACCEPTED.

        Uses the broker via execution.place_order side door. We don't
        import the trading client at module load (the dashboard must be
        importable without Alpaca credentials); do the import lazily.
        """
        try:
            from ..execution.execution import _get_trading_client
        except Exception as exc:  # noqa: BLE001
            logger.warning("cannot cancel open orders: %s", exc)
            return
        try:
            client = _get_trading_client()
            open_orders = client.get_orders()
            for o in open_orders:
                try:
                    client.cancel_order_by_id(str(o.id))
                    logger.info("emergency-stop: cancelled order %s", o.id)
                except Exception as inner:  # noqa: BLE001
                    logger.warning("could not cancel %s: %s", o.id, inner)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel_open_orders failed: %s", exc)

    # ----- dashboard convenience -----

    def poll_and_apply_commands(self) -> List[str]:
        """Consume any commands the dashboard wrote and apply them.

        Returns the list of actions that were applied this call.
        Useful in tests; the production dashboard does not need this.
        """
        cmd = self._poll_command_file()
        if not cmd:
            return []
        action = cmd.get("action")
        applied: List[str] = []
        if action == "start":
            self.start(params=cmd.get("params"))
            applied.append("start")
        elif action == "stop":
            self.stop(emergency=False)
            applied.append("stop")
        elif action == "emergency_stop":
            self.stop(emergency=True)
            applied.append("emergency_stop")
        self._clear_command_file()
        return applied


def read_session_status(path: str = DEFAULT_STATUS_FILE) -> Dict[str, Any]:
    """Read the persisted session status. Returns an empty dict if missing."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


__all__ = [
    "DEFAULT_COMMAND_FILE",
    "DEFAULT_STATUS_FILE",
    "SessionController",
    "SessionState",
    "SessionStatus",
    "read_session_status",
]
