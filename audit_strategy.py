#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_strategy.py (Patch 0003)

Strategy auditor: loads a strategy genome from SQLite and re-runs a stress suite.

Example:
  python audit_strategy.py --db strategy_results.sqlite --dataset-id demo_40000_900_1234 --rank 1
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from bt_eval import (
    BacktestCosts,
    compute_simple_returns,
    pnl_from_positions,
    metrics_from_pnl,
    metrics_from_pnl_masked,
    walkforward_metrics,
    safe_bar_seconds,
)
from feature_lib import FeatureStore
from results_db import top_strategies, load_strategy_genome


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=str, default="strategy_results.sqlite")
    p.add_argument("--dataset-id", type=str, required=True)
    p.add_argument("--eval-id", type=str, default=None)
    p.add_argument("--hash", type=str, default=None)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=0.5)
    p.add_argument("--cost-mults", type=str, default="1,2,3,5")
    p.add_argument("--delays", type=str, default="0,1")
    p.add_argument("--mdd-floor", type=float, default=0.02)
    p.add_argument("--calmar-cap", type=float, default=25.0)
    return p.parse_args()


def _split_index(n_close: int, holdout_frac: float) -> int:
    h = float(holdout_frac)
    h = min(max(h, 0.05), 0.50)
    split = int(round(n_close * (1.0 - h)))
    split = max(2, min(n_close - 2, split))
    return split


def _positions_from_alpha(alpha: np.ndarray, threshold: float, mask: np.ndarray) -> np.ndarray:
    thr = float(threshold)
    pos = np.zeros_like(alpha, dtype=np.int8)
    pos[(alpha > thr) & mask] = 1
    pos[(alpha < -thr) & mask] = -1
    return pos


def _load_dataset_from_cache(dataset_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    ts_path = os.path.join("cache", f"{dataset_id}_ts.npy")
    ohlcv_path = os.path.join("cache", f"{dataset_id}_ohlcv.npy")
    meta_path = os.path.join("cache", f"{dataset_id}_meta.json")

    if not os.path.exists(ohlcv_path):
        raise FileNotFoundError(f"Missing cache file: {ohlcv_path}")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"Missing cache file: {ts_path}")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    ts = np.load(ts_path, mmap_mode="r")
    ohlcv = np.load(ohlcv_path, mmap_mode="r")

    bar_seconds = int(meta.get("bar_seconds") or 0)
    if bar_seconds <= 0:
        bar_seconds = safe_bar_seconds(ts, default_bar_seconds=86400)

    return ts, ohlcv, meta, bar_seconds


def _eval_genome(
    genome: dict[str, Any],
    close: np.ndarray,
    volume: np.ndarray | None,
    bar_seconds: int,
    *,
    holdout_frac: float,
    costs: BacktestCosts,
    cost_mult: float,
    delay_bars: int,
    mdd_floor: float,
    calmar_cap: float,
) -> dict[str, Any]:
    close_f = close.astype(np.float64, copy=False)
    ret = compute_simple_returns(close_f)

    feats = FeatureStore(close=close_f, volume=volume, simple_ret=ret, max_cache_items=64)

    alpha = np.zeros(close_f.shape[0], dtype=np.float32)
    for part in genome["features"]:
        fname = str(part["name"])
        w = int(part["window"])
        weight = float(part["weight"])
        alpha += weight * feats.get(fname, w)

    gate = genome.get("vol_gate", {"mode": "any", "window": 80})
    gate_mask = feats.vol_gate_mask(int(gate.get("window", 80)), str(gate.get("mode", "any")))

    valid = np.isfinite(alpha)
    mask = gate_mask & valid

    pos = _positions_from_alpha(alpha, float(genome["threshold"]), mask)

    split_close = _split_index(int(close_f.shape[0]), holdout_frac)

    # train slice
    pos_train = pos[: split_close + 1]
    ret_train = ret[:split_close]
    pnl_train, trade_train = pnl_from_positions(pos_train, ret_train, costs, cost_multiplier=cost_mult)
    m_train = metrics_from_pnl(pnl_train, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
    wf = walkforward_metrics(pnl_train, bar_seconds, n_splits=4, mdd_floor=mdd_floor, calmar_cap=calmar_cap)

    # holdout slice (reset)
    pos_hold = pos[split_close:]
    ret_hold = ret[split_close:]

    d = int(max(0, delay_bars))
    if d > 0:
        pos_hold = np.roll(pos_hold, d)
        pos_hold[:d] = 0

    pnl_hold, trade_hold = pnl_from_positions(pos_hold, ret_hold, costs, cost_multiplier=cost_mult)
    m_hold = metrics_from_pnl(pnl_hold, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)

    am_train = metrics_from_pnl_masked(
        pnl_train, (pos_train[:-1] != 0), bar_seconds,
        min_bars=max(40, int(0.02 * pnl_train.size)),
        mdd_floor=mdd_floor, calmar_cap=calmar_cap
    )
    am_hold = metrics_from_pnl_masked(
        pnl_hold, (pos_hold[:-1] != 0), bar_seconds,
        min_bars=max(30, int(0.02 * pnl_hold.size)),
        mdd_floor=mdd_floor, calmar_cap=calmar_cap
    )

    return {
        "split_close": int(split_close),
        "delay_bars": int(d),
        "cost_mult": float(cost_mult),
        "train": m_train,
        "holdout": m_hold,
        "train_trade": trade_train,
        "holdout_trade": trade_hold,
        "train_active": am_train,
        "holdout_active": am_hold,
        "train_wf": wf,
    }


def main() -> int:
    args = _parse_args()

    strat_hash = args.hash
    genome = None

    if strat_hash is None and args.rank > 0:
        _eval_id, rows = top_strategies(args.db, args.dataset_id, limit=args.rank, eval_id=args.eval_id)
        if not rows or len(rows) < args.rank:
            print("No strategy found for given rank.")
            return 2
        strat_hash = rows[args.rank - 1]["strategy_hash"]
        args.eval_id = _eval_id

    if strat_hash is None:
        print("ERROR: provide --hash or --rank")
        return 2

    _eval_id, genome = load_strategy_genome(args.db, args.dataset_id, eval_id=args.eval_id, strategy_hash=strat_hash)
    if genome is None:
        print(f"Strategy not found: dataset_id={args.dataset_id} eval_id={_eval_id} hash={strat_hash}")
        return 2

    ts, ohlcv, meta, bar_seconds = _load_dataset_from_cache(args.dataset_id)
    close = ohlcv[:, 3]
    volume = ohlcv[:, 4] if ohlcv.shape[1] >= 5 else None

    print("=" * 100)
    print("AUDIT STRATEGY")
    print(f"dataset_id: {args.dataset_id}")
    print(f"eval_id(db): {_eval_id}")
    print(f"strategy_hash: {strat_hash}")
    print(f"bars: {int(meta.get('n_bars') or close.shape[0])} | bar_seconds: {bar_seconds}")
    print(f"holdout_frac(audit): {args.holdout_frac}")
    print("-" * 100)
    print("genome:")
    print(genome)
    print("=" * 100)

    costs = BacktestCosts(cost_bps=float(args.cost_bps), slippage_bps=float(args.slippage_bps))
    cost_mults = [float(x.strip()) for x in args.cost_mults.split(",") if x.strip()]
    delays = [int(x.strip()) for x in args.delays.split(",") if x.strip()]

    rows_out = []
    for cm in cost_mults:
        for d in delays:
            res = _eval_genome(
                genome,
                close=close,
                volume=volume,
                bar_seconds=bar_seconds,
                holdout_frac=float(args.holdout_frac),
                costs=costs,
                cost_mult=float(cm),
                delay_bars=int(d),
                mdd_floor=float(args.mdd_floor),
                calmar_cap=float(args.calmar_cap),
            )
            mh = res["holdout"]
            th = res["holdout_trade"]
            rows_out.append(
                (cm, d, mh.get("sharpe", 0.0), mh.get("calmar", 0.0), mh.get("total_return", 0.0), mh.get("max_drawdown", 0.0),
                 th.get("trades", 0), th.get("turnover_mean", 0.0), th.get("exposure", 0.0))
            )

    print("Holdout stress table (OOS-first):")
    print("-" * 100)
    print(f"{'costx':>6} {'delay':>5} {'sharpe':>8} {'calmar':>8} {'ret':>9} {'mdd':>8} {'tr':>6} {'turn':>7} {'exp':>6}")
    print("-" * 100)
    for cm, d, sh, ca, rt, mdd, tr, turn, exp in rows_out:
        print(f"{cm:>6.2f} {d:>5d} {float(sh):>8.3f} {float(ca):>8.3f} {float(rt):>9.3f} {float(mdd):>8.3f} {int(tr):>6d} {float(turn):>7.3f} {float(exp):>6.2f}")
    print("-" * 100)

    base_res = _eval_genome(
        genome, close=close, volume=volume, bar_seconds=bar_seconds,
        holdout_frac=float(args.holdout_frac), costs=costs,
        cost_mult=1.0, delay_bars=0,
        mdd_floor=float(args.mdd_floor), calmar_cap=float(args.calmar_cap),
    )
    print("\nDetailed metrics (costx=1.0 delay=0):")
    print("train:", base_res["train"])
    print("holdout:", base_res["holdout"])
    print("train_trade:", base_res["train_trade"])
    print("holdout_trade:", base_res["holdout_trade"])
    print("train_active:", base_res["train_active"])
    print("holdout_active:", base_res["holdout_active"])
    print("train_wf:", base_res["train_wf"])
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
