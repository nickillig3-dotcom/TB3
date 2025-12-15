#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vectorized backtest + metrics (Patch 0002).

We assume:
- close[] length N
- pos[] length N, representing position held from bar t -> t+1 (i.e. applied to return[t])
- return array has length N-1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class BacktestCosts:
    cost_bps: float = 1.0      # commission / fees
    slippage_bps: float = 0.5  # model slippage

    def total_cost_rate(self) -> float:
        # Convert bps to fraction
        return (float(self.cost_bps) + float(self.slippage_bps)) / 10_000.0


def safe_bar_seconds(ts: np.ndarray | None, default_bar_seconds: int = 86400) -> int:
    """
    Infer bar spacing from timestamps (int seconds/ms/ns) if possible.
    """
    if ts is None:
        return int(default_bar_seconds)

    try:
        if ts.size < 3:
            return int(default_bar_seconds)
        # Try to infer unit by magnitude
        med = float(np.median(np.diff(ts.astype(np.int64))))
        if med <= 0:
            return int(default_bar_seconds)

        # If it's in nanoseconds
        if med > 1e12:
            bar_sec = med / 1e9
        # milliseconds
        elif med > 1e9:
            bar_sec = med / 1e3
        else:
            bar_sec = med

        bar_sec = int(max(1, round(bar_sec)))
        # Clamp to sane range (1s..7d)
        return int(min(max(bar_sec, 1), 7 * 86400))
    except Exception:
        return int(default_bar_seconds)


def compute_simple_returns(close: np.ndarray) -> np.ndarray:
    close = close.astype(np.float64, copy=False)
    # r[t] = close[t+1]/close[t] - 1
    return (close[1:] / close[:-1]) - 1.0


def pnl_from_positions(
    pos: np.ndarray,
    simple_ret: np.ndarray,
    costs: BacktestCosts,
    *,
    cost_multiplier: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Returns pnl array length N-1.
    """
    pos = pos.astype(np.int8, copy=False)
    simple_ret = simple_ret.astype(np.float64, copy=False)

    if pos.shape[0] != simple_ret.shape[0] + 1:
        raise ValueError("pos must have length len(simple_ret)+1")

    # Trades at time t to adjust position for bar t -> t+1
    delta = np.diff(np.concatenate([np.array([0], dtype=np.int8), pos]))
    turnover = np.abs(delta).astype(np.float64)

    c = costs.total_cost_rate() * float(cost_multiplier)

    pnl = (pos[:-1].astype(np.float64) * simple_ret) - (c * turnover[:-1])

    meta = {
        "turnover_sum": float(np.sum(turnover)),
        "turnover_mean": float(np.mean(turnover)),
        "trades": int(np.sum(turnover > 0)),
        "exposure": float(np.mean(pos[:-1] != 0)),
    }
    return pnl, meta


def max_drawdown_from_equity(eq: np.ndarray) -> float:
    """
    eq: equity curve (starts at 1.0)
    Returns negative value (e.g. -0.32).
    """
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(np.min(dd))


def metrics_from_pnl(pnl: np.ndarray, bar_seconds: int) -> dict[str, Any]:
    pnl = pnl.astype(np.float64, copy=False)
    if pnl.size < 5:
        return {"error": "too_few_bars"}

    ann_factor = float(SECONDS_PER_YEAR / float(max(1, bar_seconds)))

    mu = float(np.mean(pnl))
    sig = float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0
    sharpe = (math.sqrt(ann_factor) * mu / sig) if sig > 1e-12 else 0.0

    eq = np.cumprod(1.0 + pnl)
    total_return = float(eq[-1] - 1.0)

    # CAGR approximation
    years = pnl.size / ann_factor
    if years > 0 and eq[-1] > 0:
        cagr = float(eq[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    mdd = max_drawdown_from_equity(np.concatenate([[1.0], eq]))
    calmar = float(cagr / abs(mdd)) if mdd < -1e-12 else float("inf")

    return {
        "bars": int(pnl.size),
        "bar_seconds": int(bar_seconds),
        "ann_factor": ann_factor,
        "mean_bar_return": mu,
        "std_bar_return": sig,
        "sharpe": float(sharpe),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": float(mdd),
        "calmar": calmar if math.isfinite(calmar) else 1e9,
    }


def split_segments(pnl: np.ndarray, n_splits: int) -> list[np.ndarray]:
    n = int(pnl.size)
    n_splits = int(max(1, n_splits))
    seg = n // n_splits
    if seg < 50:
        # too short → return one segment
        return [pnl]
    out = []
    for i in range(n_splits):
        a = i * seg
        b = (i + 1) * seg if i < n_splits - 1 else n
        out.append(pnl[a:b])
    return out


def walkforward_metrics(pnl: np.ndarray, bar_seconds: int, n_splits: int = 4) -> dict[str, Any]:
    segs = split_segments(pnl, n_splits=n_splits)
    seg_metrics = [metrics_from_pnl(s, bar_seconds) for s in segs]

    # Extract sharpe + total_return per segment
    seg_sharpes = [m.get("sharpe", 0.0) for m in seg_metrics if "error" not in m]
    seg_rets = [m.get("total_return", 0.0) for m in seg_metrics if "error" not in m]

    if not seg_sharpes:
        return {"error": "wf_failed"}

    pos_segments = int(sum(1 for r in seg_rets if r > 0))
    med_sharpe = float(np.median(seg_sharpes))
    std_sharpe = float(np.std(seg_sharpes)) if len(seg_sharpes) > 1 else 0.0

    # "dominance": share of pnl coming from best segment
    seg_pnl_sum = [float(np.sum(s)) for s in segs]
    total = float(sum(seg_pnl_sum))
    dominance = 1.0
    if abs(total) > 1e-12:
        dominance = float(max(seg_pnl_sum) / total) if total > 0 else float(min(seg_pnl_sum) / total)

    return {
        "n_segments": int(len(segs)),
        "positive_segments": pos_segments,
        "segment_sharpes": [float(x) for x in seg_sharpes],
        "segment_total_returns": [float(x) for x in seg_rets],
        "median_sharpe": med_sharpe,
        "std_sharpe": std_sharpe,
        "dominance": float(dominance),
    }
