#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner Research Engine (Patch 0003)

Patch 0003 upgrades:
- Train/Holdout evaluation (true OOS slice) + position reset at holdout start.
- Anti-degenerate filters: min exposure / min active bars, calmar floor (in bt_eval).
- Eval versioning via eval_id (hash of EvalConfig + score version).
- Rolling features produce NaNs early; we apply a valid mask so we don't trade on invalid bars.

Still: research-only, no execution/broker integration.
"""

from __future__ import annotations

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
    # --- Evaluation identity ---
    score_version: str = "score_v2"
    # --- Costs ---
    costs: BacktestCosts = BacktestCosts(cost_bps=1.0, slippage_bps=0.5)
    stress_cost_mult: float = 3.0

    # --- Split ---
    holdout_frac: float = 0.20  # last 20% is OOS
    min_train_bars: int = 5000
    min_holdout_bars: int = 1500

    # --- Metrics hygiene ---
    mdd_floor: float = 0.02
    calmar_cap: float = 25.0
    n_walkforward_splits: int = 4

    # --- Degeneracy / robustness filters ---
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

    # --- Performance gates ---
    min_train_sharpe: float = 0.25
    min_holdout_sharpe: float = 0.05
    min_holdout_stress_sharpe: float = 0.00

    min_wf_positive_segments: int = 2
    max_wf_dominance: float = 0.85


def eval_id_from_eval_cfg(eval_cfg: EvalConfig) -> str:
    """
    Stable eval identifier to version results in the DB.
    """
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

        genome = {
            "type": "linear_alpha_v1",
            "features": parts,
            "threshold": thr,
            "vol_gate": gate,
        }
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


def _score_v2(
    mt: dict[str, Any],
    mh: dict[str, Any],
    mh_stress: dict[str, Any],
    wf: dict[str, Any],
    tt: dict[str, Any],
    th: dict[str, Any],
) -> float:
    """
    OOS-first scoring.
    """
    train_sh = float(mt.get("sharpe", 0.0))
    hold_sh = float(mh.get("sharpe", 0.0))
    stress_sh = float(mh_stress.get("sharpe", 0.0))

    hold_cal = float(mh.get("calmar", 0.0))
    hold_mdd = abs(float(mh.get("max_drawdown", 0.0)))
    train_mdd = abs(float(mt.get("max_drawdown", 0.0)))

    gap = max(0.0, train_sh - hold_sh)

    nseg = int(wf.get("n_segments", 1))
    posseg = int(wf.get("positive_segments", 0))
    stability = posseg / max(1, nseg)

    dominance = abs(float(wf.get("dominance", 1.0)))
    turnover = float(tt.get("turnover_mean", 0.0))
    exp_h = float(th.get("exposure", 0.0))

    s = (
        0.55 * hold_sh +
        0.20 * train_sh +
        0.15 * stress_sh +
        0.10 * hold_cal
    )
    s *= (0.75 + 0.25 * stability)
    s -= 0.35 * gap
    s -= 0.22 * hold_mdd + 0.06 * (-train_mdd)
    s -= 0.05 * turnover
    s += 0.10 * min(exp_h, 0.60)
    s -= 0.15 * max(0.0, dominance - 0.60)
    return float(s)


def evaluate_genomes_batch(genomes: list[dict[str, Any]], eval_cfg: EvalConfig) -> list[dict[str, Any]]:
    """
    Executed inside worker.
    Returns list of accepted strategies with metrics + score.
    """
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

            # --- Train ---
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

            # --- Holdout (OOS) ---
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

            score = _score_v2(m_train, m_hold, m_hold_stress, wf, trade_train, trade_hold)

            strat_hash = stable_hash({"genome_v": 1, "genome": genome})
            out.append(
                {
                    "strategy_hash": strat_hash,
                    "genome": genome,
                    "metrics": metrics,
                    "score": float(score),
                    "pass_flags": "train+wf+holdout+stress",
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
