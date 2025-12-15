#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_strategy.py (Patch 0005)

Patch 0005:
- Adds time-block CV report with clean --cv-cost-mult flag
- Adds regime attribution table (vol tertiles x trend tertiles => 9 regimes)
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
    p.add_argument("--cv-cost-mult", type=float, default=3.0)

    # Regime
    p.add_argument("--regime-vol-window", type=int, default=160)
    p.add_argument("--regime-trend-window", type=int, default=160)
    p.add_argument("--regime-min-bars", type=int, default=600)
    p.add_argument("--regime-min-exp", type=float, default=0.03)

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


def _rolling_mean(x: np.ndarray, w: int, out_len: int) -> np.ndarray:
    w = int(max(1, w))
    out = np.full(int(out_len), np.nan, dtype=np.float64)
    if x.size < w:
        return out
    c = np.cumsum(np.insert(x.astype(np.float64, copy=False), 0, 0.0))
    m = (c[w:] - c[:-w]) / float(w)
    start = w
    end = min(out.size, start + m.size)
    if end > start:
        out[start:end] = m[: (end - start)]
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

    # holdout
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
        "pos": pos,
        "ret": ret,
        "pnl_train": pnl_train,
        "pnl_hold": pnl_hold,
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
    rets_c = []
    pos_folds = 0
    pos_folds_cost = 0

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
        rtc = float(m_c.get("total_return", 0.0))

        sh.append(s0); sh_d.append(s1); sh_c.append(s2)
        rets.append(rt); rets_c.append(rtc)

        if rt > 0:
            pos_folds += 1
        if rtc > 0:
            pos_folds_cost += 1

        rows.append((i + 1, a, b, s0, s1, s2, rt, rtc, float(m.get("max_drawdown", 0.0))))

    if len(sh) < 2:
        return {"error": "cv_insufficient_folds", "folds_ok": int(len(sh)), "k": int(k)}

    sh_np = np.asarray(sh, dtype=np.float64)
    sd_np = np.asarray(sh_d, dtype=np.float64)
    sc_np = np.asarray(sh_c, dtype=np.float64)

    return {
        "n_folds": int(len(sh)),
        "positive_folds": int(pos_folds),
        "positive_folds_coststress": int(pos_folds_cost),
        "median_sharpe": float(np.median(sh_np)),
        "min_sharpe": float(np.min(sh_np)),
        "std_sharpe": float(np.std(sh_np)),
        "median_sharpe_delay1": float(np.median(sd_np)),
        "min_sharpe_delay1": float(np.min(sd_np)),
        "median_sharpe_coststress": float(np.median(sc_np)),
        "min_sharpe_coststress": float(np.min(sc_np)),
        "fold_rows": rows,
        "warmup": int(warmup),
        "seg": int(seg),
        "delay_bars": int(cv_delay),
        "stress_cost_mult": float(stress_cost_mult),
    }


def _compute_regime_labels(
    close: np.ndarray,
    volume: np.ndarray | None,
    ret: np.ndarray,
    *,
    vol_window: int,
    trend_window: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_close = int(close.shape[0])
    feats = FeatureStore(close=close.astype(np.float64, copy=False), volume=volume, simple_ret=ret, max_cache_items=32)

    try:
        vol = feats.get("vol_s", int(vol_window)).astype(np.float64, copy=False)
    except Exception:
        vol = np.full(n_close, np.nan, dtype=np.float64)

    trend = _rolling_mean(ret, int(trend_window), out_len=n_close)

    def _tertile_bins(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        m = np.isfinite(x)
        bins = np.full(x.shape[0], -1, dtype=np.int8)
        if m.sum() < 200:
            return bins, float("nan"), float("nan")
        q1, q2 = np.nanquantile(x[m], [1/3, 2/3])
        bins[(x <= q1) & m] = 0
        bins[(x > q1) & (x <= q2) & m] = 1
        bins[(x > q2) & m] = 2
        return bins, float(q1), float(q2)

    vb, vq1, vq2 = _tertile_bins(vol)
    tb, tq1, tq2 = _tertile_bins(trend)

    reg = np.full(n_close, -1, dtype=np.int8)
    ok = (vb >= 0) & (tb >= 0)
    reg[ok] = (vb[ok] * 3 + tb[ok]).astype(np.int8)

    info = {
        "vol_window": int(vol_window),
        "trend_window": int(trend_window),
        "vol_q33": vq1, "vol_q66": vq2,
        "trend_q33": tq1, "trend_q66": tq2,
        "bars_unknown": int((reg < 0).sum()),
    }
    return reg, info


def _regime_report(
    pos: np.ndarray,
    pnl_full: np.ndarray,
    regime: np.ndarray,
    bar_seconds: int,
    *,
    min_bars: int,
    min_exposure: float,
) -> dict[str, Any]:
    n_close = int(pos.shape[0])
    if pnl_full.shape[0] != n_close - 1:
        return {"error": "shape_mismatch"}

    reg = regime[:-1]
    ok = (reg >= 0)
    pos_use = pos[:-1]
    active = (pos_use != 0)

    ann_factor = (365.25 * 24.0 * 3600.0) / float(max(1, bar_seconds))
    sqrt_ann = float(math.sqrt(ann_factor))

    rows = []
    total_pos = 0.0
    pos_contribs = []
    coverage = 0
    worst_sh = None

    for rid in range(9):
        m = (reg == rid) & ok
        bars = int(m.sum())
        if bars <= 0:
            rows.append((rid, rid // 3, rid % 3, 0, 0, 0.0, 0.0, 0.0))
            continue
        act = int((active & m).sum())
        exp = float(act) / float(max(1, bars))
        pnl = pnl_full[m]
        pnl_sum = float(np.sum(pnl))
        pnl_mean = pnl_sum / float(max(1, bars))
        pnl_std = float(np.std(pnl)) if pnl.size > 1 else 0.0
        sh = float((pnl_mean / pnl_std) * sqrt_ann) if pnl_std > 1e-12 else 0.0

        if bars >= int(min_bars) and exp >= float(min_exposure):
            coverage += 1
            worst_sh = sh if worst_sh is None else min(worst_sh, sh)

        pc = max(0.0, pnl_sum)
        pos_contribs.append(pc)
        total_pos += pc

        rows.append((rid, rid // 3, rid % 3, bars, act, exp, pnl_sum, sh))

    dom = float(max(pos_contribs) / total_pos) if total_pos > 1e-12 and pos_contribs else 0.0
    if worst_sh is None:
        worst_sh = 0.0

    return {
        "coverage": int(coverage),
        "dominance": float(dom),
        "worst_sharpe": float(worst_sh),
        "rows": rows,
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
    print("AUDIT STRATEGY (Patch 0005)")
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
    base_bundle = None
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
            if cm == 1.0 and d == 0:
                base_bundle = res
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
        stress_cost_mult=float(args.cv_cost_mult),
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
        pfc = f"{cv['positive_folds_coststress']}/{cv['n_folds']}"
        print(f"  folds={cv['n_folds']} pos_folds={pf} pos_cost_folds={pfc} warmup={cv['warmup']} seg={cv['seg']} delay={cv['delay_bars']} cost_mult={cv['stress_cost_mult']}")
        print(f"  cv_median_sharpe={cv['median_sharpe']:.3f}  cv_min_sharpe={cv['min_sharpe']:.3f}  cv_std_sharpe={cv['std_sharpe']:.3f}")
        print(f"  cv_med_delay1={cv['median_sharpe_delay1']:.3f}  cv_min_delay1={cv['min_sharpe_delay1']:.3f}")
        print(f"  cv_med_coststress={cv['median_sharpe_coststress']:.3f}  cv_min_coststress={cv['min_sharpe_coststress']:.3f}")
        print("-" * 110)
        print(f"{'fold':>4} {'a':>7} {'b':>7} {'sh':>8} {'sh_d1':>8} {'sh_cost':>8} {'ret':>9} {'retC':>9} {'mdd':>8}")
        print("-" * 110)
        for (fi, a, b, sh0, sh1, sh2, rt, rtc, mdd) in cv["fold_rows"]:
            print(f"{fi:>4d} {a:>7d} {b:>7d} {sh0:>8.3f} {sh1:>8.3f} {sh2:>8.3f} {rt:>9.3f} {rtc:>9.3f} {mdd:>8.3f}")
        print("-" * 110)

    # Detailed base metrics
    if base_bundle is None:
        base_bundle = _eval_holdout_suite(
            genome, close=close, volume=volume, bar_seconds=bar_seconds,
            holdout_frac=float(args.holdout_frac), costs=costs,
            cost_mult=1.0, delay_bars=0,
            mdd_floor=float(args.mdd_floor), calmar_cap=float(args.calmar_cap),
        )

    print("\nDetailed metrics (costx=1.0 delay=0):")
    print("train:", base_bundle["train"])
    print("holdout:", base_bundle["holdout"])
    print("train_trade:", base_bundle["train_trade"])
    print("holdout_trade:", base_bundle["holdout_trade"])
    print("train_active:", base_bundle["train_active"])
    print("holdout_active:", base_bundle["holdout_active"])
    print("train_wf:", base_bundle["train_wf"])

    # Regime attribution (base pnl full)
    pos_full = base_bundle["pos"]
    ret_full = base_bundle["ret"]
    pnl_full = np.concatenate([base_bundle["pnl_train"], base_bundle["pnl_hold"]])

    reg, reg_info = _compute_regime_labels(
        close=close.astype(np.float64, copy=False),
        volume=volume.astype(np.float64, copy=False) if volume is not None else None,
        ret=ret_full,
        vol_window=int(args.regime_vol_window),
        trend_window=int(args.regime_trend_window),
    )
    rep = _regime_report(
        pos=pos_full,
        pnl_full=pnl_full,
        regime=reg,
        bar_seconds=bar_seconds,
        min_bars=int(args.regime_min_bars),
        min_exposure=float(args.regime_min_exp),
    )

    print("\nRegime attribution (base, full series):")
    if "error" in rep:
        print("  regime error:", rep)
    else:
        print(f"  regimes=9 coverage={rep['coverage']}/9 dominance={rep['dominance']:.3f} worst_sharpe={rep['worst_sharpe']:.3f}")
        print(f"  vol_window={reg_info.get('vol_window')} trend_window={reg_info.get('trend_window')} bars_unknown={reg_info.get('bars_unknown')}")
        print("-" * 110)
        print(f"{'rid':>3} {'vol':>3} {'trd':>3} {'bars':>7} {'act':>7} {'exp':>6} {'pnl_sum':>10} {'sharpe':>8}")
        print("-" * 110)
        for (rid, vb, tb, bars, act, exp, pnl_sum, sh) in rep["rows"]:
            print(f"{rid:>3d} {vb:>3d} {tb:>3d} {bars:>7d} {act:>7d} {exp:>6.2f} {pnl_sum:>10.3f} {sh:>8.3f}")
        print("-" * 110)

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
