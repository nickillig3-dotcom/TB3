#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner Research Engine (Patch 0002)

Core properties:
- Multiprocessing with spawn-safe worker init
- Dataset loaded via memmap so each worker has read-only view without copying
- Random strategy generator in a meaningful (but still simple) search space
- Hierarchical filters: fast metrics -> walk-forward stability -> stress costs
- SQLite persistence of the best candidates

This is v0 infrastructure. The "weapon quality" comes from later patches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from sm_utils import stable_hash, utc_now_iso
from bt_eval import BacktestCosts, compute_simple_returns, pnl_from_positions, metrics_from_pnl, walkforward_metrics
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

    # Columns: open, high, low, close, volume
    _G_CLOSE = ohlcv[:, 3].astype(np.float64, copy=False)
    vol = ohlcv[:, 4] if ohlcv.shape[1] >= 5 else None
    _G_VOLUME = vol.astype(np.float64, copy=False) if vol is not None else None

    _G_RET = compute_simple_returns(_G_CLOSE)
    _G_BAR_SECONDS = int(bar_seconds)
    _G_FEATS = FeatureStore(close=_G_CLOSE, volume=_G_VOLUME, simple_ret=_G_RET, max_cache_items=24)


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
    costs: BacktestCosts = BacktestCosts(cost_bps=1.0, slippage_bps=0.5)
    stress_cost_mult: float = 3.0
    n_walkforward_splits: int = 4

    min_trades: int = 20
    max_turnover_mean: float = 1.2  # avg abs(delta_pos) per bar
    max_drawdown_abs: float = 0.75

    min_sharpe: float = 0.20
    min_wf_positive_segments: int = 2
    max_wf_dominance: float = 0.85
    min_stress_sharpe: float = 0.05


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
# Evaluation (inside worker)
# -----------------------------

def _positions_from_alpha(alpha: np.ndarray, threshold: float, mask: np.ndarray) -> np.ndarray:
    thr = float(threshold)
    pos = np.zeros_like(alpha, dtype=np.int8)
    pos[(alpha > thr) & mask] = 1
    pos[(alpha < -thr) & mask] = -1
    return pos


def _score(metrics_base: dict[str, Any], wf: dict[str, Any], trade: dict[str, Any]) -> float:
    sharpe = float(metrics_base.get("sharpe", 0.0))
    calmar = float(metrics_base.get("calmar", 0.0))
    mdd = float(metrics_base.get("max_drawdown", 0.0))
    turnover = float(trade.get("turnover_mean", 0.0))
    pos_segments = int(wf.get("positive_segments", 0))
    n_segments = int(wf.get("n_segments", 1))

    stability = pos_segments / max(1, n_segments)
    s = 0.70 * sharpe + 0.30 * calmar
    s *= (0.50 + 0.50 * stability)
    s -= 0.25 * abs(mdd)
    s -= 0.05 * turnover
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

    out: list[dict[str, Any]] = []

    for genome in genomes:
        try:
            if genome.get("type") != "linear_alpha_v1":
                continue

            alpha = np.zeros_like(close, dtype=np.float32)
            for part in genome["features"]:
                fname = str(part["name"])
                w = int(part["window"])
                weight = float(part["weight"])
                alpha += weight * feats.get(fname, w)

            gate = genome.get("vol_gate", {"mode": "any", "window": 80})
            gate_mode = str(gate.get("mode", "any"))
            gate_w = int(gate.get("window", 80))
            mask = feats.vol_gate_mask(gate_w, gate_mode)

            pos = _positions_from_alpha(alpha, genome["threshold"], mask)

            # base run
            pnl, trade = pnl_from_positions(pos, ret, eval_cfg.costs, cost_multiplier=1.0)
            m_base = metrics_from_pnl(pnl, bar_seconds)
            if "error" in m_base:
                continue

            # fast filters
            if trade["trades"] < eval_cfg.min_trades:
                continue
            if trade["turnover_mean"] > eval_cfg.max_turnover_mean:
                continue
            if abs(m_base["max_drawdown"]) > eval_cfg.max_drawdown_abs:
                continue
            if m_base["sharpe"] < eval_cfg.min_sharpe:
                continue

            wf = walkforward_metrics(pnl, bar_seconds, n_splits=eval_cfg.n_walkforward_splits)
            if "error" in wf:
                continue
            if wf["positive_segments"] < eval_cfg.min_wf_positive_segments:
                continue
            if abs(wf["dominance"]) > eval_cfg.max_wf_dominance:
                continue

            # stress costs
            pnl_s, _trade_s = pnl_from_positions(pos, ret, eval_cfg.costs, cost_multiplier=eval_cfg.stress_cost_mult)
            m_stress = metrics_from_pnl(pnl_s, bar_seconds)
            if "error" in m_stress:
                continue
            if m_stress["sharpe"] < eval_cfg.min_stress_sharpe:
                continue

            metrics = {
                "base": m_base,
                "stress": m_stress,
                "wf": wf,
                "trade": trade,
            }
            score = _score(m_base, wf, trade)

            strat_hash = stable_hash({"dataset": "v1", "genome": genome})
            out.append(
                {
                    "strategy_hash": strat_hash,
                    "genome": genome,
                    "metrics": metrics,
                    "score": float(score),
                    "pass_flags": "base+wf+stress",
                }
            )
        except Exception:
            # ignore broken genomes
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
    """
    Main orchestrator.
    """
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
            # fill pipeline
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

        # drain remaining quickly
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
