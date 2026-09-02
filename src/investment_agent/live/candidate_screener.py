"""Deterministic candidate screener.

WHAT
====
Given a list of symbols and their bar histories, rank and filter
them using a transparent, deterministic scoring rule so the LLM
sees only the top-N candidates. The screener never calls the LLM;
it only consumes price/volume data and returns the short list.

WHY
====
The full LLM call costs Featherless credits. A naive 7-agent
multi-LLM dispatch across hundreds of symbols would burn the
$25 budget inside one decision interval. The screener collapses
the universe to the top N (default 3) so the LLM only sees the
candidates that actually have a chance of producing a clean
trade.

HOW
====
``CandidateScreener.screen(symbol_data)`` takes a
``Dict[symbol, pd.DataFrame]`` of OHLCV bars and returns a list
of ``ScreenResult`` sorted by score (highest first). The score
is a weighted blend of:

  * momentum    : (close[-1] - close[-20]) / close[-20]
  * volatility  : stdev(returns[-20:]) * sqrt(252)
  * volume      : mean(volume[-20:]) relative to the universe
  * trend_abs   : |momentum|

Each component is normalized to [0, 1] within the universe before
the weighted blend, so the final score is also in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScreenResult:
    """Single candidate returned by the screener."""
    symbol: str
    score: float
    momentum: float
    volatility: float
    relative_volume: float
    last_price: float
    n_bars: int
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "score": float(self.score),
            "momentum": float(self.momentum),
            "volatility": float(self.volatility),
            "relative_volume": float(self.relative_volume),
            "last_price": float(self.last_price),
            "n_bars": int(self.n_bars),
            "reason": self.reason,
        }


@dataclass
class CandidateScreener:
    """Deterministic candidate screener.

    Parameters
    ----------
    top_n : int
        Maximum number of candidates returned.
    min_bars : int
        Minimum number of bars required to score a symbol.
    momentum_weight : float
        Weight of the momentum component in [0, 1].
    volatility_weight : float
        Weight of the volatility component in [0, 1].
    volume_weight : float
        Weight of the volume component in [0, 1].
    min_relative_volume : float
        Reject candidates with relative volume below this cutoff.
    """
    top_n: int = 3
    min_bars: int = 20
    momentum_weight: float = 0.4
    volatility_weight: float = 0.3
    volume_weight: float = 0.3
    min_relative_volume: float = 0.0

    def screen(
        self, symbol_data: Dict[str, pd.DataFrame]
    ) -> List[ScreenResult]:
        """Screen the universe and return up to ``top_n`` candidates.

        Parameters
        ----------
        symbol_data : Dict[symbol, DataFrame]
            Each DataFrame must have a ``close`` column and at
            least ``min_bars`` rows. A ``volume`` column is
            preferred but optional.
        """
        per_symbol: List[Dict[str, float]] = []
        for sym, df in symbol_data.items():
            if df is None or len(df) < self.min_bars:
                continue
            try:
                closes = df["close"].astype(float).values
                last_price = float(closes[-1])
                if last_price <= 0:
                    continue
                # 20-day momentum
                ref_price = float(closes[-self.min_bars])
                if ref_price <= 0:
                    continue
                momentum = (last_price - ref_price) / ref_price
                # 20-day volatility (annualized)
                rets = np.diff(closes[-self.min_bars:]) / closes[-self.min_bars:-1]
                if len(rets) < 2:
                    continue
                vol = float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0
                # Mean volume (fall back to 1.0 if not present)
                if "volume" in df.columns:
                    vol_mean = float(df["volume"].astype(float).tail(self.min_bars).mean())
                else:
                    vol_mean = 1.0
            except Exception:
                continue
            per_symbol.append({
                "symbol": sym,
                "last_price": last_price,
                "momentum": momentum,
                "volatility": vol,
                "volume": vol_mean,
                "n_bars": len(df),
            })

        if not per_symbol:
            return []

        # Normalize momentum and volatility within the universe.
        mags = [abs(d["momentum"]) for d in per_symbol]
        max_mag = max(mags) or 1.0
        vols = [d["volatility"] for d in per_symbol]
        max_vol = max(vols) or 1.0
        raw_vols = [d["volume"] for d in per_symbol]
        max_raw_vol = max(raw_vols) or 1.0

        scored: List[ScreenResult] = []
        for d in per_symbol:
            momentum_norm = abs(d["momentum"]) / max_mag
            vol_norm = d["volatility"] / max_vol
            rel_vol = d["volume"] / max_raw_vol
            if rel_vol < self.min_relative_volume:
                continue
            score = (
                self.momentum_weight * momentum_norm
                + self.volatility_weight * vol_norm
                + self.volume_weight * rel_vol
            )
            scored.append(ScreenResult(
                symbol=d["symbol"],
                score=float(score),
                momentum=float(d["momentum"]),
                volatility=float(d["volatility"]),
                relative_volume=float(rel_vol),
                last_price=float(d["last_price"]),
                n_bars=int(d["n_bars"]),
                reason=(
                    f"mom={d['momentum']:+.2%} vol={d['volatility']:.2f} "
                    f"rvol={rel_vol:.2f}"
                ),
            ))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:self.top_n]


__all__ = ["CandidateScreener", "ScreenResult"]
