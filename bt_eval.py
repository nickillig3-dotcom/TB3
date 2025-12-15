#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vectorized backtest + metrics (Patch 0003).

Conventions:
- close[] length N
- pos[] length N, where pos[t] is the position held during return[t] (bar t -> t+1)
- simple_ret[] length N-1, simple_ret[t] = close[t+1]/close[t] - 1

Key goals for research robustness:
- Calmar computed with drawdown floor (prevents "flat strategy" exploit).
- Optional masked/active metrics helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SECONDS_PER_YEAR = 365.25 * 24 * 3600
_EPS = 1e-12


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
        return int(min(max(bar_sec, 1), 7 * 86400))
    except Exception:
        return int(default_bar_seconds)


def compute_simple_returns(close: np.ndarray) -> np.ndarray:
    close = close.astype(np.float64, copy=False)
    return (close[1:] / close[:-1]) - 1.0


def pnl_from_positions(
    pos: np.ndarray,
    simple_ret: np.ndarray,
    costs: BacktestCosts,
    *,
    cost_multiplier: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Returns:
      pnl: length N-1
      trade_meta: trades/turnover/exposure/active_bars
    """
    pos = pos.astype(np.int8, copy=False)
    simple_ret = simple_ret.astype(np.float64, copy=False)

    if pos.shape[0] != simple_ret.shape[0] + 1:
        raise ValueError("pos must have length len(simple_ret)+1")

    # delta position at each bar (including initial entry from flat)
    delta = np.diff(np.concatenate([np.array([0], dtype=np.int8), pos]))
    turnover = np.abs(delta).astype(np.float64)

    c = costs.total_cost_rate() * float(cost_multiplier)

    pnl = (pos[:-1].astype(np.float64) * simple_ret) - (c * turnover[:-1])

    active_mask = (pos[:-1] != 0)
    trade_meta = {
        "turnover_sum": float(np.sum(turnover)),
        "turnover_mean": float(np.mean(turnover)),
        "trades": int(np.sum(turnover > 0)),
        "active_bars": int(np.sum(active_mask)),
        "exposure": float(np.mean(active_mask)),
    }
    return pnl, trade_meta


def max_drawdown_from_equity(eq: np.ndarray) -> float:
    """
    eq: equity curve (starts at 1.0)
    Returns negative value (e.g. -0.32).
    """
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(np.min(dd))


def _ann_factor(bar_seconds: int) -> float:
    return float(SECONDS_PER_YEAR / float(max(1, bar_seconds)))


def metrics_from_pnl(
    pnl: np.ndarray,
    bar_seconds: int,
    *,
    mdd_floor: float = 0.02,
    calmar_cap: float = 25.0,
) -> dict[str, Any]:
    """
    Base metrics for a pnl series.

    mdd_floor: prevents calmar blow-up when drawdown is near zero.
    calmar_cap: absolute cap for calmar (keeps scores sane).
    """
    pnl = pnl.astype(np.float64, copy=False)
    if pnl.size < 5:
        return {"error": "too_few_bars"}

    ann_factor = _ann_factor(int(bar_seconds))

    mu = float(np.mean(pnl))
    sig = float(np.std(pnl, ddof=1)) if pnl.size > 1 else 0.0
    sharpe = (math.sqrt(ann_factor) * mu / sig) if sig > 1e-12 else 0.0

    eq = np.cumprod(1.0 + pnl)
    total_return = float(eq[-1] - 1.0)

    years = pnl.size / ann_factor
    if years > 0 and eq[-1] > 0:
        cagr = float(eq[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    mdd = max_drawdown_from_equity(np.concatenate([[1.0], eq]))

    denom = max(abs(float(mdd)), float(mdd_floor))
    calmar = float(cagr / denom)

    # keep sane
    if not math.isfinite(calmar):
        calmar = 0.0
    calmar = float(max(-abs(calmar_cap), min(abs(calmar_cap), calmar)))

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
        "calmar": float(calmar),
    }


def metrics_from_pnl_masked(
    pnl: np.ndarray,
    mask: np.ndarray,
    bar_seconds: int,
    *,
    min_bars: int = 50,
    mdd_floor: float = 0.02,
    calmar_cap: float = 25.0,
) -> dict[str, Any]:
    """
    Metrics on masked subset of pnl bars (e.g. active bars).
    """
    mask = mask.astype(bool, copy=False)
    if mask.shape[0] != pnl.shape[0]:
        raise ValueError("mask must match pnl length")

    sub = pnl[mask]
    if sub.size < int(min_bars):
        return {"error": "too_few_masked_bars", "masked_bars": int(sub.size)}

    m = metrics_from_pnl(sub, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
    m["masked_bars"] = int(sub.size)
    return m


def split_segments(pnl: np.ndarray, n_splits: int) -> list[np.ndarray]:
    n = int(pnl.size)
    n_splits = int(max(1, n_splits))
    seg = n // n_splits
    if seg < 200:
        return [pnl]
    out = []
    for i in range(n_splits):
        a = i * seg
        b = (i + 1) * seg if i < n_splits - 1 else n
        out.append(pnl[a:b])
    return out


def walkforward_metrics(
    pnl: np.ndarray,
    bar_seconds: int,
    *,
    n_splits: int = 4,
    mdd_floor: float = 0.02,
    calmar_cap: float = 25.0,
) -> dict[str, Any]:
    segs = split_segments(pnl, n_splits=n_splits)
    seg_metrics = [metrics_from_pnl(s, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap) for s in segs]

    seg_sharpes = [m.get("sharpe", 0.0) for m in seg_metrics if "error" not in m]
    seg_rets = [m.get("total_return", 0.0) for m in seg_metrics if "error" not in m]

    if not seg_sharpes:
        return {"error": "wf_failed"}

    pos_segments = int(sum(1 for r in seg_rets if r > 0))
    med_sharpe = float(np.median(seg_sharpes))
    std_sharpe = float(np.std(seg_sharpes)) if len(seg_sharpes) > 1 else 0.0

    # dominance: share of pnl coming from best segment (avoid single lucky block)
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
