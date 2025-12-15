#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_strategy.py (Patch 0004)

Patch 0004:
- Adds time-block CV report (same idea as engine Deep Validation Stage).
- Still prints holdout stress table (cost multipliers + delay).

Example:
  python audit_strategy.py --db strategy_results.sqlite --dataset-id demo_40000_900_1234 --rank 1 --cv-folds 5
"""

from __future__ import annotations

import argparse
import json
import os
import math
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

    # CV
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-delay", type=int, default=1)
    p.add_argument("--cv-min-fold-bars", type=int, default=800)

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


def _warmup_index_for_genome(genome: dict[str, Any]) -> int:
    try:
        ws = [int(p.get("window", 1)) for p in genome.get("features", [])]
        gate = genome.get("vol_gate", {}) or {}
        ws.append(int(gate.get("window", 1)))
        wmax = max(ws) if ws else 1
        return int(max(2, min(5000, wmax + 3)))
    except Exception:
        return 50


def _apply_delay(pos_seg: np.ndarray, delay_bars: int) -> np.ndarray:
    d = int(max(0, delay_bars))
    if d == 0:
        return pos_seg
    out = np.roll(pos_seg, d).copy()
    out[:d] = 0
    return out


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


def _build_alpha_pos(
    genome: dict[str, Any],
    close: np.ndarray,
    volume: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close_f = close.astype(np.float64, copy=False)
    ret = compute_simple_returns(close_f)
    feats = FeatureStore(close=close_f, volume=volume, simple_ret=ret, max_cache_items=128)

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
    return alpha, pos, ret


def _eval_holdout_suite(
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
    _, pos, ret = _build_alpha_pos(genome, close, volume)
    n_close = int(pos.shape[0])
    split_close = _split_index(n_close, holdout_frac)

    # train
    pos_train = pos[: split_close + 1]
    ret_train = ret[:split_close]
    pnl_train, trade_train = pnl_from_positions(pos_train, ret_train, costs, cost_multiplier=cost_mult)
    m_train = metrics_from_pnl(pnl_train, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
    wf = walkforward_metrics(pnl_train, bar_seconds, n_splits=4, mdd_floor=mdd_floor, calmar_cap=calmar_cap)

    # holdout (reset)
    pos_hold = pos[split_close:]
    ret_hold = ret[split_close:]

    pos_hold = _apply_delay(pos_hold, int(delay_bars))

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
        "delay_bars": int(delay_bars),
        "cost_mult": float(cost_mult),
        "train": m_train,
        "holdout": m_hold,
        "train_trade": trade_train,
        "holdout_trade": trade_hold,
        "train_active": am_train,
        "holdout_active": am_hold,
        "train_wf": wf,
    }


def _timeblock_cv_report(
    genome: dict[str, Any],
    close: np.ndarray,
    volume: np.ndarray | None,
    bar_seconds: int,
    *,
    costs: BacktestCosts,
    stress_cost_mult: float,
    cv_folds: int,
    cv_delay: int,
    cv_min_fold_bars: int,
    mdd_floor: float,
    calmar_cap: float,
) -> dict[str, Any]:
    _, pos, ret = _build_alpha_pos(genome, close, volume)

    n_close = int(pos.shape[0])
    warmup = _warmup_index_for_genome(genome)
    start = int(min(max(0, warmup), n_close - 2))
    end = n_close

    n_eff = end - start
    k = int(max(2, cv_folds))
    if n_eff < k * int(cv_min_fold_bars):
        k = max(2, n_eff // int(cv_min_fold_bars))
    if k < 2:
        return {"error": "cv_too_short"}

    seg = n_eff // k
    if seg < int(cv_min_fold_bars):
        return {"error": "cv_fold_too_short", "seg": int(seg), "k": int(k), "n_eff": int(n_eff)}

    rows = []
    sh = []
    sh_d = []
    sh_c = []
    rets = []
    pos_folds = 0

    for i in range(k):
        a = start + i * seg
        b = start + (i + 1) * seg if i < k - 1 else end

        if b - a < int(cv_min_fold_bars):
            continue

        pos_seg = pos[a:b]
        ret_seg = ret[a : (b - 1)]

        pnl, _ = pnl_from_positions(pos_seg, ret_seg, costs, cost_multiplier=1.0)
        m = metrics_from_pnl(pnl, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
        pnl_d, _ = pnl_from_positions(_apply_delay(pos_seg, cv_delay), ret_seg, costs, cost_multiplier=1.0)
        m_d = metrics_from_pnl(pnl_d, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
        pnl_c, _ = pnl_from_positions(pos_seg, ret_seg, costs, cost_multiplier=float(stress_cost_mult))
        m_c = metrics_from_pnl(pnl_c, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)

        if "error" in m or "error" in m_d or "error" in m_c:
            continue

        s0 = float(m.get("sharpe", 0.0))
        s1 = float(m_d.get("sharpe", 0.0))
        s2 = float(m_c.get("sharpe", 0.0))
        rt = float(m.get("total_return", 0.0))

        sh.append(s0); sh_d.append(s1); sh_c.append(s2); rets.append(rt)
        if rt > 0:
            pos_folds += 1

        rows.append((i + 1, a, b, s0, s1, s2, rt, float(m.get("max_drawdown", 0.0))))

    if len(sh) < 2:
        return {"error": "cv_insufficient_folds", "folds_ok": int(len(sh)), "k": int(k)}

    sh_np = np.asarray(sh, dtype=np.float64)
    sd_np = np.asarray(sh_d, dtype=np.float64)
    sc_np = np.asarray(sh_c, dtype=np.float64)

    return {
        "n_folds": int(len(sh)),
        "positive_folds": int(pos_folds),
        "median_sharpe": float(np.median(sh_np)),
        "min_sharpe": float(np.min(sh_np)),
        "std_sharpe": float(np.std(sh_np)),
        "median_sharpe_delay1": float(np.median(sd_np)),
        "median_sharpe_coststress": float(np.median(sc_np)),
        "fold_rows": rows,
        "warmup": int(warmup),
        "seg": int(seg),
        "delay_bars": int(cv_delay),
        "stress_cost_mult": float(stress_cost_mult),
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
    print("AUDIT STRATEGY (Patch 0004)")
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

    # Holdout stress table
    rows_out = []
    for cm in cost_mults:
        for d in delays:
            res = _eval_holdout_suite(
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

    # CV report
    cv = _timeblock_cv_report(
        genome,
        close=close,
        volume=volume,
        bar_seconds=bar_seconds,
        costs=costs,
        stress_cost_mult=float(args.cost_mults.split(",")[2]) if "," in args.cost_mults else 3.0,
        cv_folds=int(args.cv_folds),
        cv_delay=int(args.cv_delay),
        cv_min_fold_bars=int(args.cv_min_fold_bars),
        mdd_floor=float(args.mdd_floor),
        calmar_cap=float(args.calmar_cap),
    )
    print("\nTime-block CV report:")
    if "error" in cv:
        print("  CV error:", cv)
    else:
        pf = f"{cv['positive_folds']}/{cv['n_folds']}"
        print(f"  folds={cv['n_folds']} positive_folds={pf} warmup={cv['warmup']} seg={cv['seg']} delay={cv['delay_bars']}")
        print(f"  cv_median_sharpe={cv['median_sharpe']:.3f}  cv_min_sharpe={cv['min_sharpe']:.3f}  cv_std_sharpe={cv['std_sharpe']:.3f}")
        print(f"  cv_med_delay1={cv['median_sharpe_delay1']:.3f}  cv_med_coststress={cv['median_sharpe_coststress']:.3f}")
        print("-" * 100)
        print(f"{'fold':>4} {'a':>7} {'b':>7} {'sh':>8} {'sh_d1':>8} {'sh_cost':>8} {'ret':>9} {'mdd':>8}")
        print("-" * 100)
        for (fi, a, b, sh0, sh1, sh2, rt, mdd) in cv["fold_rows"]:
            print(f"{fi:>4d} {a:>7d} {b:>7d} {sh0:>8.3f} {sh1:>8.3f} {sh2:>8.3f} {rt:>9.3f} {mdd:>8.3f}")
        print("-" * 100)

    # Detailed base metrics (costx=1, delay=0)
    base_res = _eval_holdout_suite(
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
