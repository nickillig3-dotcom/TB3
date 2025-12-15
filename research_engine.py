#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner Research Engine (Patch 0004)

Patch 0004 upgrades:
- Deep Validation Stage (DVS): time-block cross-validation (k folds) across (warmup..end)
- Stress integrated into CV:
    * base: delay=0, cost_mult=1
    * delay stress: delay=cv_delay_bars, cost_mult=1
    * cost stress: delay=0, cost_mult=stress_cost_mult
- Score v3: CV-first, reduces "last-holdout lucky" domination.
- No DB schema changes; CV summary stored in metrics_json["cv"].

Research-only, no execution integration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from sm_utils import stable_hash, utc_now_iso
from bt_eval import (
    BacktestCosts,
    compute_simple_returns,
    pnl_from_positions,
    metrics_from_pnl,
    metrics_from_pnl_masked,
    walkforward_metrics,
)
from feature_lib import FeatureStore


# -----------------------------
# Worker globals (spawn-safe)
# -----------------------------
_G_CLOSE: np.ndarray | None = None
_G_VOLUME: np.ndarray | None = None
_G_RET: np.ndarray | None = None
_G_BAR_SECONDS: int = 86400
_G_FEATS: FeatureStore | None = None


def _worker_init(ts_path: str, ohlcv_path: str, bar_seconds: int) -> None:
    """
    Worker initializer: load dataset memmaps once per process.
    """
    global _G_CLOSE, _G_VOLUME, _G_RET, _G_BAR_SECONDS, _G_FEATS

    ohlcv = np.load(ohlcv_path, mmap_mode="r")
    _G_CLOSE = ohlcv[:, 3].astype(np.float64, copy=False)
    vol = ohlcv[:, 4] if ohlcv.shape[1] >= 5 else None
    _G_VOLUME = vol.astype(np.float64, copy=False) if vol is not None else None

    _G_RET = compute_simple_returns(_G_CLOSE)
    _G_BAR_SECONDS = int(bar_seconds)
    _G_FEATS = FeatureStore(close=_G_CLOSE, volume=_G_VOLUME, simple_ret=_G_RET, max_cache_items=32)


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class SearchSpace:
    features: tuple[str, ...] = ("mom_ir", "mr_z", "vol_s", "volu_z")
    windows: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320)
    n_features_min: int = 2
    n_features_max: int = 4
    weight_abs_max: float = 2.5
    threshold_min: float = 0.15
    threshold_max: float = 1.5
    vol_gate_modes: tuple[str, ...] = ("any", "high", "low")
    vol_gate_windows: tuple[int, ...] = (40, 80, 160)


@dataclass(frozen=True)
class EvalConfig:
    # --- Identity ---
    score_version: str = "score_v3_cv"

    # --- Costs ---
    costs: BacktestCosts = BacktestCosts(cost_bps=1.0, slippage_bps=0.5)
    stress_cost_mult: float = 3.0

    # --- Split (Stage-1) ---
    holdout_frac: float = 0.20
    min_train_bars: int = 5000
    min_holdout_bars: int = 1500

    # --- Metrics hygiene ---
    mdd_floor: float = 0.02
    calmar_cap: float = 25.0
    n_walkforward_splits: int = 4

    # --- Degeneracy / robustness filters (Stage-1) ---
    min_exposure_train: float = 0.02
    min_exposure_holdout: float = 0.01
    min_active_bars_train: int = 300
    min_active_bars_holdout: int = 80

    min_trades_train: int = 60
    min_trades_holdout: int = 10

    max_turnover_mean_train: float = 1.2
    max_turnover_mean_holdout: float = 1.5
    max_drawdown_abs_train: float = 0.75
    max_drawdown_abs_holdout: float = 0.75

    # --- Performance gates (Stage-1) ---
    min_train_sharpe: float = 0.25
    min_holdout_sharpe: float = 0.05
    min_holdout_stress_sharpe: float = 0.00
    min_wf_positive_segments: int = 2
    max_wf_dominance: float = 0.85

    # --- Deep Validation Stage (Stage-2 CV) ---
    deep_validation: bool = True
    cv_folds: int = 5
    cv_delay_bars: int = 1

    # fold geometry
    cv_min_fold_bars: int = 800

    # gates
    cv_min_positive_folds_frac: float = 0.60
    cv_min_positive_folds: int = 0  # if >0 overrides frac rule
    cv_min_sharpe: float = -0.20
    cv_min_median_sharpe_base: float = 0.20
    cv_min_median_sharpe_delay1: float = 0.10
    cv_min_median_sharpe_coststress: float = 0.05


def eval_id_from_eval_cfg(eval_cfg: EvalConfig) -> str:
    payload = asdict(eval_cfg)
    h = stable_hash(payload)
    return f"eval_{h[:12]}"


@dataclass
class ResearchConfig:
    dataset_id: str
    ts_path: str
    ohlcv_path: str
    bar_seconds: int
    db_path: str = "strategy_results.sqlite"
    run_minutes: float = 1.0
    workers: int = 4
    batch_size: int = 96
    max_in_flight: int = 8
    seed: int = 1234
    mode: str = "research"
    search_space: SearchSpace = SearchSpace()
    eval_cfg: EvalConfig = EvalConfig()


# -----------------------------
# Strategy generator
# -----------------------------

def generate_genomes(rng: np.random.Generator, n: int, space: SearchSpace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    feats = list(space.features)
    wins = list(space.windows)

    for _ in range(int(n)):
        k = int(rng.integers(space.n_features_min, space.n_features_max + 1))
        chosen_idx = rng.choice(len(feats), size=k, replace=False)
        chosen = [feats[i] for i in chosen_idx]

        parts = []
        for name in chosen:
            w = int(rng.choice(wins))
            weight = float(rng.uniform(-space.weight_abs_max, space.weight_abs_max))
            parts.append({"name": name, "window": w, "weight": weight})

        thr = float(rng.uniform(space.threshold_min, space.threshold_max))
        gate_mode = str(rng.choice(list(space.vol_gate_modes)))
        gate = {"mode": gate_mode, "window": int(rng.choice(list(space.vol_gate_windows)))}

        genome = {"type": "linear_alpha_v1", "features": parts, "threshold": thr, "vol_gate": gate}
        out.append(genome)
    return out


# -----------------------------
# Evaluation helpers (inside worker)
# -----------------------------

def _positions_from_alpha(alpha: np.ndarray, threshold: float, mask: np.ndarray) -> np.ndarray:
    thr = float(threshold)
    pos = np.zeros_like(alpha, dtype=np.int8)
    pos[(alpha > thr) & mask] = 1
    pos[(alpha < -thr) & mask] = -1
    return pos


def _split_index(n_close: int, holdout_frac: float) -> int:
    h = float(holdout_frac)
    if h <= 0.0:
        return n_close
    h = min(max(h, 0.05), 0.50)
    split = int(round(n_close * (1.0 - h)))
    split = max(2, min(n_close - 2, split))
    return split


def _warmup_index_for_genome(genome: dict[str, Any]) -> int:
    """
    Approx warmup to avoid evaluating long initial NaN/invalid region.
    """
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


def _timeblock_cv(
    pos: np.ndarray,
    ret: np.ndarray,
    bar_seconds: int,
    *,
    genome: dict[str, Any],
    costs: BacktestCosts,
    stress_cost_mult: float,
    folds: int,
    delay_bars: int,
    min_fold_bars: int,
    mdd_floor: float,
    calmar_cap: float,
) -> dict[str, Any]:
    """
    Returns CV summary + small fold vectors.
    Uses close-index folds on range [warmup..N).
    """
    n_close = int(pos.shape[0])
    if ret.shape[0] != n_close - 1:
        return {"error": "cv_shape_mismatch"}

    warmup = _warmup_index_for_genome(genome)
    start = int(min(max(0, warmup), n_close - 2))
    end = n_close

    n_eff = end - start
    k = int(max(2, folds))
    if n_eff < k * int(min_fold_bars):
        # Reduce folds automatically if series is short.
        k = max(2, n_eff // int(min_fold_bars))
    if k < 2:
        return {"error": "cv_too_short"}

    seg = n_eff // k
    if seg < int(min_fold_bars):
        return {"error": "cv_fold_too_short", "seg": int(seg), "k": int(k), "n_eff": int(n_eff)}

    fold_sh = []
    fold_sh_d1 = []
    fold_sh_cost = []
    fold_ret = []
    fold_start_end = []

    pos_folds = 0

    for i in range(k):
        a = start + i * seg
        b = start + (i + 1) * seg if i < k - 1 else end
        if b - a < int(min_fold_bars):
            continue
        if b - a < 6:
            continue

        pos_seg = pos[a:b]
        ret_seg = ret[a : (b - 1)]

        # base
        pnl, _trade = pnl_from_positions(pos_seg, ret_seg, costs, cost_multiplier=1.0)
        m = metrics_from_pnl(pnl, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
        if "error" in m:
            continue

        # delay stress
        pos_d = _apply_delay(pos_seg, int(delay_bars))
        pnl_d, _ = pnl_from_positions(pos_d, ret_seg, costs, cost_multiplier=1.0)
        m_d = metrics_from_pnl(pnl_d, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
        if "error" in m_d:
            continue

        # cost stress
        pnl_c, _ = pnl_from_positions(pos_seg, ret_seg, costs, cost_multiplier=float(stress_cost_mult))
        m_c = metrics_from_pnl(pnl_c, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
        if "error" in m_c:
            continue

        sh = float(m.get("sharpe", 0.0))
        sh_d = float(m_d.get("sharpe", 0.0))
        sh_c = float(m_c.get("sharpe", 0.0))
        rt = float(m.get("total_return", 0.0))

        fold_sh.append(sh)
        fold_sh_d1.append(sh_d)
        fold_sh_cost.append(sh_c)
        fold_ret.append(rt)
        fold_start_end.append((int(a), int(b)))

        if rt > 0:
            pos_folds += 1

    if len(fold_sh) < 2:
        return {"error": "cv_insufficient_folds", "folds_ok": int(len(fold_sh)), "k": int(k)}

    fold_sh_np = np.asarray(fold_sh, dtype=np.float64)
    fold_sh_d_np = np.asarray(fold_sh_d1, dtype=np.float64)
    fold_sh_c_np = np.asarray(fold_sh_cost, dtype=np.float64)

    summary = {
        "n_folds": int(len(fold_sh)),
        "fold_start_end": fold_start_end,
        "positive_folds": int(pos_folds),
        "median_sharpe": float(np.median(fold_sh_np)),
        "min_sharpe": float(np.min(fold_sh_np)),
        "std_sharpe": float(np.std(fold_sh_np)) if fold_sh_np.size > 1 else 0.0,
        "median_sharpe_delay1": float(np.median(fold_sh_d_np)),
        "median_sharpe_coststress": float(np.median(fold_sh_c_np)),
        "fold_sharpes": [float(x) for x in fold_sh],
        "fold_sharpes_delay1": [float(x) for x in fold_sh_d1],
        "fold_sharpes_coststress": [float(x) for x in fold_sh_cost],
        "fold_total_returns": [float(x) for x in fold_ret],
        "warmup": int(warmup),
        "delay_bars": int(delay_bars),
        "stress_cost_mult": float(stress_cost_mult),
    }
    return summary


def _score_v3(
    m_train: dict[str, Any],
    m_hold: dict[str, Any],
    m_hold_stress: dict[str, Any],
    wf: dict[str, Any],
    trade_train: dict[str, Any],
    trade_hold: dict[str, Any],
    cv: dict[str, Any],
) -> float:
    """
    CV-first score. The aim is to reduce "one block lucky" dominance.
    """
    train_sh = float(m_train.get("sharpe", 0.0))
    hold_sh = float(m_hold.get("sharpe", 0.0))
    stress_sh = float(m_hold_stress.get("sharpe", 0.0))

    gap = max(0.0, train_sh - hold_sh)

    cv_med = float(cv.get("median_sharpe", 0.0))
    cv_min = float(cv.get("min_sharpe", 0.0))
    cv_std = float(cv.get("std_sharpe", 0.0))
    cv_med_d = float(cv.get("median_sharpe_delay1", 0.0))
    cv_med_c = float(cv.get("median_sharpe_coststress", 0.0))

    pos_folds = int(cv.get("positive_folds", 0))
    n_folds = int(max(1, cv.get("n_folds", 1)))
    pos_frac = pos_folds / n_folds

    dominance = abs(float(wf.get("dominance", 1.0)))
    turnover = float(trade_train.get("turnover_mean", 0.0))
    hold_mdd = abs(float(m_hold.get("max_drawdown", 0.0)))

    s = (
        0.55 * cv_med +
        0.20 * cv_min +
        0.12 * cv_med_d +
        0.08 * cv_med_c +
        0.05 * stress_sh +
        0.05 * hold_sh
    )
    s *= (0.85 + 0.15 * pos_frac)
    s -= 0.22 * cv_std
    s -= 0.22 * gap
    s -= 0.14 * hold_mdd
    s -= 0.05 * turnover
    s -= 0.12 * max(0.0, dominance - 0.60)
    return float(s)


def evaluate_genomes_batch(genomes: list[dict[str, Any]], eval_cfg: EvalConfig) -> list[dict[str, Any]]:
    global _G_CLOSE, _G_RET, _G_BAR_SECONDS, _G_FEATS
    if _G_CLOSE is None or _G_RET is None or _G_FEATS is None:
        raise RuntimeError("Worker not initialized with dataset")

    close = _G_CLOSE
    ret = _G_RET
    bar_seconds = int(_G_BAR_SECONDS)
    feats = _G_FEATS

    n_close = int(close.shape[0])
    split_close = _split_index(n_close, eval_cfg.holdout_frac)

    train_ret = ret[: max(0, min(split_close, ret.shape[0]))]
    holdout_ret = ret[max(0, min(split_close, ret.shape[0])) :]

    if train_ret.size < int(eval_cfg.min_train_bars):
        return []
    if holdout_ret.size < int(eval_cfg.min_holdout_bars):
        return []

    out: list[dict[str, Any]] = []

    for genome in genomes:
        try:
            if genome.get("type") != "linear_alpha_v1":
                continue

            alpha = np.zeros(n_close, dtype=np.float32)
            for part in genome["features"]:
                fname = str(part["name"])
                w = int(part["window"])
                weight = float(part["weight"])
                alpha += weight * feats.get(fname, w)

            gate = genome.get("vol_gate", {"mode": "any", "window": 80})
            gate_mode = str(gate.get("mode", "any"))
            gate_w = int(gate.get("window", 80))
            gate_mask = feats.vol_gate_mask(gate_w, gate_mode)

            valid = np.isfinite(alpha)
            mask = gate_mask & valid

            pos = _positions_from_alpha(alpha, genome["threshold"], mask)

            # --- Stage 1: Train ---
            pos_train = pos[: split_close + 1]
            pnl_train, trade_train = pnl_from_positions(pos_train, train_ret, eval_cfg.costs, cost_multiplier=1.0)
            m_train = metrics_from_pnl(
                pnl_train,
                bar_seconds,
                mdd_floor=eval_cfg.mdd_floor,
                calmar_cap=eval_cfg.calmar_cap,
            )
            if "error" in m_train:
                continue

            if trade_train["exposure"] < eval_cfg.min_exposure_train:
                continue
            if trade_train["active_bars"] < eval_cfg.min_active_bars_train:
                continue
            if trade_train["trades"] < eval_cfg.min_trades_train:
                continue
            if trade_train["turnover_mean"] > eval_cfg.max_turnover_mean_train:
                continue
            if abs(m_train["max_drawdown"]) > eval_cfg.max_drawdown_abs_train:
                continue
            if m_train["sharpe"] < eval_cfg.min_train_sharpe:
                continue

            wf = walkforward_metrics(
                pnl_train,
                bar_seconds,
                n_splits=eval_cfg.n_walkforward_splits,
                mdd_floor=eval_cfg.mdd_floor,
                calmar_cap=eval_cfg.calmar_cap,
            )
            if "error" in wf:
                continue
            if wf["positive_segments"] < eval_cfg.min_wf_positive_segments:
                continue
            if abs(wf["dominance"]) > eval_cfg.max_wf_dominance:
                continue

            active_mask_train = (pos_train[:-1] != 0)
            m_train_active = metrics_from_pnl_masked(
                pnl_train,
                active_mask_train,
                bar_seconds,
                min_bars=max(50, int(0.02 * pnl_train.size)),
                mdd_floor=eval_cfg.mdd_floor,
                calmar_cap=eval_cfg.calmar_cap,
            )

            # --- Stage 1: Holdout (OOS) ---
            pos_hold = pos[split_close:]
            pnl_hold, trade_hold = pnl_from_positions(pos_hold, holdout_ret, eval_cfg.costs, cost_multiplier=1.0)
            m_hold = metrics_from_pnl(
                pnl_hold,
                bar_seconds,
                mdd_floor=eval_cfg.mdd_floor,
                calmar_cap=eval_cfg.calmar_cap,
            )
            if "error" in m_hold:
                continue

            if trade_hold["exposure"] < eval_cfg.min_exposure_holdout:
                continue
            if trade_hold["active_bars"] < eval_cfg.min_active_bars_holdout:
                continue
            if trade_hold["trades"] < eval_cfg.min_trades_holdout:
                continue
            if trade_hold["turnover_mean"] > eval_cfg.max_turnover_mean_holdout:
                continue
            if abs(m_hold["max_drawdown"]) > eval_cfg.max_drawdown_abs_holdout:
                continue
            if m_hold["sharpe"] < eval_cfg.min_holdout_sharpe:
                continue

            pnl_hold_s, _ = pnl_from_positions(
                pos_hold,
                holdout_ret,
                eval_cfg.costs,
                cost_multiplier=eval_cfg.stress_cost_mult,
            )
            m_hold_stress = metrics_from_pnl(
                pnl_hold_s,
                bar_seconds,
                mdd_floor=eval_cfg.mdd_floor,
                calmar_cap=eval_cfg.calmar_cap,
            )
            if "error" in m_hold_stress:
                continue
            if m_hold_stress["sharpe"] < eval_cfg.min_holdout_stress_sharpe:
                continue

            active_mask_hold = (pos_hold[:-1] != 0)
            m_hold_active = metrics_from_pnl_masked(
                pnl_hold,
                active_mask_hold,
                bar_seconds,
                min_bars=max(40, int(0.02 * pnl_hold.size)),
                mdd_floor=eval_cfg.mdd_floor,
                calmar_cap=eval_cfg.calmar_cap,
            )

            metrics = {
                "train": m_train,
                "holdout": m_hold,
                "holdout_stress": m_hold_stress,
                "train_wf": wf,
                "train_trade": trade_train,
                "holdout_trade": trade_hold,
                "train_active": m_train_active,
                "holdout_active": m_hold_active,
                "split": {
                    "holdout_frac": float(eval_cfg.holdout_frac),
                    "split_close": int(split_close),
                    "n_close": int(n_close),
                },
            }

            # --- Stage 2: Deep Validation (Time-block CV) ---
            pass_flags = "train+wf+holdout+stress"
            cv_summary = None

            if bool(eval_cfg.deep_validation) and int(eval_cfg.cv_folds) >= 2:
                cv_summary = _timeblock_cv(
                    pos=pos,
                    ret=ret,
                    bar_seconds=bar_seconds,
                    genome=genome,
                    costs=eval_cfg.costs,
                    stress_cost_mult=eval_cfg.stress_cost_mult,
                    folds=int(eval_cfg.cv_folds),
                    delay_bars=int(eval_cfg.cv_delay_bars),
                    min_fold_bars=int(eval_cfg.cv_min_fold_bars),
                    mdd_floor=eval_cfg.mdd_floor,
                    calmar_cap=eval_cfg.calmar_cap,
                )
                if "error" in cv_summary:
                    continue

                # gates
                n_folds_ok = int(cv_summary.get("n_folds", 0))
                pos_folds = int(cv_summary.get("positive_folds", 0))

                min_pos = int(eval_cfg.cv_min_positive_folds)
                if min_pos <= 0:
                    min_pos = int(math.ceil(float(eval_cfg.cv_min_positive_folds_frac) * float(n_folds_ok)))
                min_pos = max(1, min(min_pos, n_folds_ok))

                if pos_folds < min_pos:
                    continue

                if float(cv_summary.get("min_sharpe", -1e9)) < float(eval_cfg.cv_min_sharpe):
                    continue
                if float(cv_summary.get("median_sharpe", -1e9)) < float(eval_cfg.cv_min_median_sharpe_base):
                    continue
                if float(cv_summary.get("median_sharpe_delay1", -1e9)) < float(eval_cfg.cv_min_median_sharpe_delay1):
                    continue
                if float(cv_summary.get("median_sharpe_coststress", -1e9)) < float(eval_cfg.cv_min_median_sharpe_coststress):
                    continue

                metrics["cv"] = cv_summary
                pass_flags = "train+wf+holdout+stress+cv"

            # score (v3 uses cv if available; otherwise fallback to holdout-centric)
            if cv_summary is not None and "error" not in cv_summary:
                score = _score_v3(m_train, m_hold, m_hold_stress, wf, trade_train, trade_hold, cv_summary)
            else:
                # fallback (rare; only when deep_validation disabled)
                score = 0.55 * float(m_hold.get("sharpe", 0.0)) + 0.25 * float(m_train.get("sharpe", 0.0)) - 0.15 * max(0.0, float(m_train.get("sharpe", 0.0)) - float(m_hold.get("sharpe", 0.0)))

            strat_hash = stable_hash({"genome_v": 1, "genome": genome})
            out.append(
                {
                    "strategy_hash": strat_hash,
                    "genome": genome,
                    "metrics": metrics,
                    "score": float(score),
                    "pass_flags": pass_flags,
                }
            )
        except Exception:
            continue

    return out


# -----------------------------
# Main loop (in main process)
# -----------------------------

ProgressCallback = Callable[[dict[str, Any]], None]
ResultCallback = Callable[[dict[str, Any]], None]


def run_research(
    cfg: ResearchConfig,
    *,
    on_result: ResultCallback,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed))

    deadline = time.time() + float(cfg.run_minutes) * 60.0
    total_tested = 0
    total_accepted = 0
    best_score = -1e9
    t0 = time.time()
    last_report = t0

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=int(cfg.workers),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(cfg.ts_path, cfg.ohlcv_path, int(cfg.bar_seconds)),
    ) as ex:
        pending = set()
        max_in_flight = int(max(1, cfg.max_in_flight))

        while time.time() < deadline:
            while len(pending) < max_in_flight and time.time() < deadline:
                genomes = generate_genomes(rng, cfg.batch_size, cfg.search_space)
                fut = ex.submit(evaluate_genomes_batch, genomes, cfg.eval_cfg)
                pending.add(fut)
                total_tested += len(genomes)

            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    results = fut.result()
                except Exception:
                    continue

                for r in results:
                    total_accepted += 1
                    best_score = max(best_score, float(r.get("score", -1e9)))
                    on_result(r)

            now = time.time()
            if on_progress is not None and (now - last_report) >= 2.5:
                dt = max(1e-9, now - t0)
                on_progress(
                    {
                        "ts": utc_now_iso(),
                        "tested": total_tested,
                        "accepted": total_accepted,
                        "tested_per_sec": total_tested / dt,
                        "accepted_pct": (total_accepted / max(1, total_tested)) * 100.0,
                        "best_score": best_score,
                        "in_flight": len(pending),
                    }
                )
                last_report = now

        if pending:
            done, _ = wait(pending, timeout=5.0)
            for fut in done:
                try:
                    results = fut.result()
                except Exception:
                    continue
                for r in results:
                    total_accepted += 1
                    best_score = max(best_score, float(r.get("score", -1e9)))
                    on_result(r)

    dt = max(1e-9, time.time() - t0)
    return {
        "tested": total_tested,
        "accepted": total_accepted,
        "tested_per_sec": total_tested / dt,
        "accepted_pct": (total_accepted / max(1, total_tested)) * 100.0,
        "best_score": best_score,
        "seconds": dt,
    }
