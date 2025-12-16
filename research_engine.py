#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner Research Engine (Patch 0005)

Patch 0005:
- Evolutionary search (ElitePool + mutation + crossover + seed queue)
- CV gates v2 (min sharpe delay/cost + positive folds under cost stress)
- Regime attribution (vol tertiles x trend tertiles => 9 regimes) stored in metrics_json["regime"]
- Score v4 (CV-first + regime balance penalty)
- No DB schema changes.
"""

from __future__ import annotations

import math
import time
import copy
from dataclasses import dataclass, asdict, field
from typing import Any, Callable
from collections import OrderedDict, Counter

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
_G_REGIME_CACHE: dict[tuple[int, int], dict[str, Any]] | None = None


def _worker_init(ts_path: str, ohlcv_path: str, bar_seconds: int) -> None:
    """
    Worker initializer: load dataset memmaps once per process.
    """
    global _G_CLOSE, _G_VOLUME, _G_RET, _G_BAR_SECONDS, _G_FEATS, _G_REGIME_CACHE

    ohlcv = np.load(ohlcv_path, mmap_mode="r")
    _G_CLOSE = ohlcv[:, 3].astype(np.float64, copy=False)
    vol = ohlcv[:, 4] if ohlcv.shape[1] >= 5 else None
    _G_VOLUME = vol.astype(np.float64, copy=False) if vol is not None else None

    _G_RET = compute_simple_returns(_G_CLOSE)
    _G_BAR_SECONDS = int(bar_seconds)
    _G_FEATS = FeatureStore(close=_G_CLOSE, volume=_G_VOLUME, simple_ret=_G_RET, max_cache_items=48)
    _G_REGIME_CACHE = {}  # lazy per (vol_window, trend_window)


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
    score_version: str = "score_v4_cv_regime"

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
    cv_min_fold_bars: int = 800

    # v1 gates
    cv_min_positive_folds_frac: float = 0.60
    cv_min_positive_folds: int = 0  # if >0 overrides frac rule
    cv_min_sharpe: float = -0.20
    cv_min_median_sharpe_base: float = 0.20
    cv_min_median_sharpe_delay1: float = 0.10
    cv_min_median_sharpe_coststress: float = 0.05

    # v2 gates (new)
    cv_min_sharpe_delay1: float = -0.10
    cv_min_sharpe_coststress: float = -0.10
    cv_min_positive_folds_coststress_frac: float = 0.60
    cv_min_positive_folds_coststress: int = 0

    # --- Regime attribution (new) ---
    regime_enabled: bool = True
    regime_vol_window: int = 160
    regime_trend_window: int = 160
    regime_min_bars: int = 600
    regime_min_exposure: float = 0.03


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

    # --- Search controls (Patch 0005) ---
    search_mode: str = "evo"          # "random" | "evo"
    elite_size: int = 512
    elite_seed_genomes: list[dict[str, Any]] = field(default_factory=list)
    seen_max: int = 200_000

    evo_frac_random: float = 0.35
    evo_frac_mutate: float = 0.50
    evo_frac_crossover: float = 0.15


# -----------------------------
# Search / Evo
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _genome_fingerprint(genome: dict[str, Any]) -> int:
    """
    Cheap in-run fingerprint to avoid duplicates. Not persisted.
    """
    try:
        parts = genome.get("features", []) or []
        parts_t = tuple(sorted(
            (str(p.get("name")), int(p.get("window")), round(float(p.get("weight")), 4))
            for p in parts
        ))
        gate = genome.get("vol_gate", {}) or {}
        fp = (
            str(genome.get("type")),
            parts_t,
            round(float(genome.get("threshold")), 4),
            str(gate.get("mode", "any")),
            int(gate.get("window", 0)),
        )
        return hash(fp)
    except Exception:
        return hash(("bad",))


class ElitePool:
    def __init__(self, max_size: int, rng: np.random.Generator):
        self.max_size = int(max(8, max_size))
        self.rng = rng
        self.items: list[tuple[float, dict[str, Any], int]] = []  # (score, genome, fp)
        self._fps: set[int] = set()

    def add(self, score: float, genome: dict[str, Any]) -> None:
        fp = _genome_fingerprint(genome)
        if fp in self._fps:
            return
        self.items.append((float(score), genome, fp))
        self._fps.add(fp)
        self.items.sort(key=lambda x: x[0], reverse=True)
        if len(self.items) > self.max_size:
            # drop tail
            for _, _, fpt in self.items[self.max_size:]:
                self._fps.discard(fpt)
            self.items = self.items[: self.max_size]

    def size(self) -> int:
        return int(len(self.items))

    def best_score(self) -> float:
        return float(self.items[0][0]) if self.items else -1e9

    def sample(self) -> dict[str, Any] | None:
        if not self.items:
            return None
        n = len(self.items)
        # rank weights: top gets most
        w = np.arange(n, 0, -1, dtype=np.float64)
        p = w / w.sum()
        idx = int(self.rng.choice(n, p=p))
        return self.items[idx][1]


class CandidateGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        space: SearchSpace,
        *,
        mode: str,
        elite_size: int,
        seen_max: int,
        frac_random: float,
        frac_mutate: float,
        frac_crossover: float,
        seed_genomes: list[dict[str, Any]] | None = None,
    ):
        self.rng = rng
        self.space = space
        self.mode = str(mode).lower().strip()
        self.elite = ElitePool(max_size=int(elite_size), rng=rng)

        self.seen_max = int(max(10_000, seen_max))
        self.seen = OrderedDict()  # fp -> None (insertion ordered)

        # mixture
        s = max(1e-9, float(frac_random) + float(frac_mutate) + float(frac_crossover))
        self.frac_random = float(frac_random) / s
        self.frac_mutate = float(frac_mutate) / s
        self.frac_crossover = float(frac_crossover) / s

        # seed queue: evaluated early
        self.seed_queue: list[dict[str, Any]] = list(seed_genomes or [])

    def _seen_add(self, fp: int) -> bool:
        if fp in self.seen:
            return False
        self.seen[fp] = None
        if len(self.seen) > self.seen_max:
            self.seen.popitem(last=False)
        return True

    def _random_genome(self) -> dict[str, Any]:
        return generate_genomes(self.rng, 1, self.space)[0]

    def _mutate(self, parent: dict[str, Any]) -> dict[str, Any]:
        g = copy.deepcopy(parent)
        wins = list(self.space.windows)
        feats = list(self.space.features)

        # weight jitter
        if self.rng.random() < 0.95:
            for p in g.get("features", []):
                w = float(p.get("weight", 0.0))
                w += float(self.rng.normal(0.0, 0.20))
                p["weight"] = _clamp(w, -self.space.weight_abs_max, self.space.weight_abs_max)

        # threshold jitter
        if self.rng.random() < 0.75:
            thr = float(g.get("threshold", 0.5))
            thr += float(self.rng.normal(0.0, 0.08))
            g["threshold"] = _clamp(thr, self.space.threshold_min, self.space.threshold_max)

        # window mutate
        if self.rng.random() < 0.45 and g.get("features"):
            i = int(self.rng.integers(0, len(g["features"])))
            w0 = int(g["features"][i].get("window", wins[0]))
            if w0 in wins:
                j = wins.index(w0)
                j2 = int(_clamp(j + int(self.rng.choice([-1, 1])), 0, len(wins) - 1))
                g["features"][i]["window"] = int(wins[j2])
            else:
                g["features"][i]["window"] = int(self.rng.choice(wins))

        # feature swap (rare)
        if self.rng.random() < 0.18 and g.get("features"):
            i = int(self.rng.integers(0, len(g["features"])))
            existing = {str(p.get("name")) for p in g["features"]}
            cand = [f for f in feats if f not in existing]
            if cand:
                g["features"][i]["name"] = str(self.rng.choice(cand))

        # gate mutate
        if self.rng.random() < 0.25:
            gate = g.get("vol_gate", {}) or {}
            if self.rng.random() < 0.5:
                gate["mode"] = str(self.rng.choice(list(self.space.vol_gate_modes)))
            if self.rng.random() < 0.7:
                gate["window"] = int(self.rng.choice(list(self.space.vol_gate_windows)))
            g["vol_gate"] = gate

        # keep type
        g["type"] = "linear_alpha_v1"
        return g

    def _crossover(self, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        wins = list(self.space.windows)
        feats = list(self.space.features)

        # merge (name, window) keys
        def parts_map(g: dict[str, Any]) -> dict[tuple[str, int], float]:
            m = {}
            for p in g.get("features", []):
                k = (str(p.get("name")), int(p.get("window")))
                m[k] = float(p.get("weight", 0.0))
            return m

        ma = parts_map(a)
        mb = parts_map(b)
        keys = list(set(ma.keys()) | set(mb.keys()))
        if not keys:
            return self._random_genome()

        k = int(self.rng.integers(self.space.n_features_min, self.space.n_features_max + 1))
        if len(keys) < k:
            # pad with random keys
            while len(keys) < k:
                keys.append((str(self.rng.choice(feats)), int(self.rng.choice(wins))))
        keys = list(set(keys))
        self.rng.shuffle(keys)
        keys = keys[:k]

        child_parts = []
        for (name, w) in keys:
            wa = ma.get((name, w), None)
            wb = mb.get((name, w), None)

            # IMPORTANT: If we padded keys with random (name, window), it may exist in neither parent.
            if wa is None and wb is None:
                weight = float(self.rng.uniform(-self.space.weight_abs_max, self.space.weight_abs_max)) * 0.6
            elif wa is not None and wb is not None:
                weight = 0.5 * (wa + wb) + float(self.rng.normal(0.0, 0.10))
            elif wa is not None:
                weight = wa + float(self.rng.normal(0.0, 0.10))
            else:
                weight = wb + float(self.rng.normal(0.0, 0.10))

            weight = _clamp(weight, -self.space.weight_abs_max, self.space.weight_abs_max)
            child_parts.append({"name": name, "window": int(w), "weight": float(weight)})

        thr = 0.5 * (float(a.get("threshold", 0.5)) + float(b.get("threshold", 0.5)))
        thr += float(self.rng.normal(0.0, 0.06))
        thr = _clamp(thr, self.space.threshold_min, self.space.threshold_max)

        gate = (a.get("vol_gate") if self.rng.random() < 0.5 else b.get("vol_gate")) or {"mode": "any", "window": 80}
        # small gate mutate
        if self.rng.random() < 0.20:
            gate = dict(gate)
            if self.rng.random() < 0.5:
                gate["mode"] = str(self.rng.choice(list(self.space.vol_gate_modes)))
            if self.rng.random() < 0.7:
                gate["window"] = int(self.rng.choice(list(self.space.vol_gate_windows)))

        return {"type": "linear_alpha_v1", "features": child_parts, "threshold": float(thr), "vol_gate": gate}

    def next_batch(self, n: int) -> list[dict[str, Any]]:
        n = int(max(1, n))
        out: list[dict[str, Any]] = []

        # consume seeds first (evaluate old elites under new eval_cfg)
        while self.seed_queue and len(out) < n:
            g = self.seed_queue.pop(0)
            fp = _genome_fingerprint(g)
            if self._seen_add(fp):
                out.append(g)

        if self.mode == "random":
            while len(out) < n:
                g = self._random_genome()
                fp = _genome_fingerprint(g)
                if self._seen_add(fp):
                    out.append(g)
            return out

        # evo
        max_tries = 25 * n
        tries = 0
        while len(out) < n and tries < max_tries:
            tries += 1
            u = float(self.rng.random())

            if u < self.frac_random or self.elite.size() < 8:
                g = self._random_genome()
            elif u < self.frac_random + self.frac_mutate:
                parent = self.elite.sample()
                g = self._mutate(parent) if parent is not None else self._random_genome()
            else:
                p1 = self.elite.sample()
                p2 = self.elite.sample()
                if p1 is None or p2 is None:
                    g = self._random_genome()
                else:
                    g = self._crossover(p1, p2)

            fp = _genome_fingerprint(g)
            if self._seen_add(fp):
                out.append(g)

        # fallback fill
        while len(out) < n:
            g = self._random_genome()
            fp = _genome_fingerprint(g)
            if self._seen_add(fp):
                out.append(g)

        return out

    def update_elite(self, score: float, genome: dict[str, Any]) -> None:
        self.elite.add(float(score), genome)


# -----------------------------
# Strategy generator (random base)
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
    m = (c[w:] - c[:-w]) / float(w)  # length x.size - w + 1
    # align: mean of last w returns ends at ret index t-1 => close index t
    start = w
    end = start + m.size
    if end > out.size:
        end = out.size
        m = m[: (end - start)]
    if start < out.size and m.size > 0:
        out[start:end] = m
    return out


def _get_or_build_regime_cache(vol_window: int, trend_window: int) -> dict[str, Any]:
    global _G_REGIME_CACHE, _G_FEATS, _G_RET, _G_CLOSE
    assert _G_REGIME_CACHE is not None
    key = (int(vol_window), int(trend_window))
    if key in _G_REGIME_CACHE:
        return _G_REGIME_CACHE[key]

    if _G_FEATS is None or _G_RET is None or _G_CLOSE is None:
        _G_REGIME_CACHE[key] = {"regime": None, "info": {"error": "no_data"}}
        return _G_REGIME_CACHE[key]

    n_close = int(_G_CLOSE.shape[0])

    try:
        vol = _G_FEATS.get("vol_s", int(vol_window)).astype(np.float64, copy=False)
    except Exception:
        vol = np.full(n_close, np.nan, dtype=np.float64)

    trend = _rolling_mean(_G_RET, int(trend_window), out_len=n_close)

    # quantile bins
    def _tertile_bins(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        x = x.astype(np.float64, copy=False)
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

    regime = np.full(n_close, -1, dtype=np.int8)
    ok = (vb >= 0) & (tb >= 0)
    regime[ok] = (vb[ok] * 3 + tb[ok]).astype(np.int8)

    info = {
        "vol_window": int(vol_window),
        "trend_window": int(trend_window),
        "vol_q33": vq1, "vol_q66": vq2,
        "trend_q33": tq1, "trend_q66": tq2,
        "bars_unknown": int((regime < 0).sum()),
    }
    _G_REGIME_CACHE[key] = {"regime": regime, "info": info}
    return _G_REGIME_CACHE[key]


def _regime_attribution(
    pos: np.ndarray,
    pnl_full: np.ndarray,
    regime: np.ndarray,
    bar_seconds: int,
    *,
    min_bars: int,
    min_exposure: float,
) -> dict[str, Any]:
    """
    Compute contribution + exposure per regime over full series.
    regime is per close index (len = n_close), pnl_full is per return (len=n_close-1).
    """
    n_close = int(pos.shape[0])
    if pnl_full.shape[0] != n_close - 1:
        return {"error": "regime_shape_mismatch"}

    reg = regime[:-1]
    m_ok = (reg >= 0)
    pos_use = pos[:-1]
    active = (pos_use != 0)

    ann_factor = (365.25 * 24.0 * 3600.0) / float(max(1, bar_seconds))
    sqrt_ann = float(math.sqrt(ann_factor))

    rows = []
    pos_pnl_sums = []
    total_pos_pnl = 0.0

    coverage = 0
    worst_sh = None

    for rid in range(9):
        mask = (reg == rid) & m_ok
        bars = int(mask.sum())
        if bars <= 0:
            rows.append({
                "rid": int(rid),
                "vol_bin": int(rid // 3),
                "trend_bin": int(rid % 3),
                "bars": 0,
                "active_bars": 0,
                "exposure": 0.0,
                "pnl_sum": 0.0,
                "pnl_mean": 0.0,
                "sharpe": 0.0,
            })
            continue

        active_bars = int((active & mask).sum())
        exposure = float(active_bars) / float(max(1, bars))

        pnl_slice = pnl_full[mask]
        pnl_sum = float(np.sum(pnl_slice))
        pnl_mean = pnl_sum / float(max(1, bars))
        pnl_std = float(np.std(pnl_slice)) if pnl_slice.size > 1 else 0.0
        sharpe = float((pnl_mean / pnl_std) * sqrt_ann) if pnl_std > 1e-12 else 0.0

        if bars >= int(min_bars) and exposure >= float(min_exposure):
            coverage += 1
            if worst_sh is None:
                worst_sh = sharpe
            else:
                worst_sh = min(worst_sh, sharpe)

        pos_contrib = max(0.0, pnl_sum)
        pos_pnl_sums.append(pos_contrib)
        total_pos_pnl += pos_contrib

        rows.append({
            "rid": int(rid),
            "vol_bin": int(rid // 3),
            "trend_bin": int(rid % 3),
            "bars": int(bars),
            "active_bars": int(active_bars),
            "exposure": float(exposure),
            "pnl_sum": float(pnl_sum),
            "pnl_mean": float(pnl_mean),
            "sharpe": float(sharpe),
        })

    dominance = float(max(pos_pnl_sums) / total_pos_pnl) if total_pos_pnl > 1e-12 and pos_pnl_sums else 0.0
    if worst_sh is None:
        worst_sh = 0.0

    return {
        "n_regimes": 9,
        "coverage": int(coverage),
        "dominance": float(dominance),
        "worst_sharpe": float(worst_sh),
        "rows": rows,
    }


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
    Returns CV summary + fold vectors.
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
        k = max(2, n_eff // int(min_fold_bars))
    if k < 2:
        return {"error": "cv_too_short"}

    seg = n_eff // k
    if seg < int(min_fold_bars):
        return {"error": "cv_fold_too_short", "seg": int(seg), "k": int(k), "n_eff": int(n_eff)}

    fold_sh = []
    fold_sh_d1 = []
    fold_sh_cost = []
    fold_rt = []
    fold_rt_d1 = []
    fold_rt_cost = []
    fold_start_end = []

    pos_folds = 0
    pos_folds_cost = 0

    for i in range(k):
        a = start + i * seg
        b = start + (i + 1) * seg if i < k - 1 else end
        if b - a < int(min_fold_bars):
            continue
        if b - a < 6:
            continue

        pos_seg = pos[a:b]
        ret_seg = ret[a : (b - 1)]

        pnl, _ = pnl_from_positions(pos_seg, ret_seg, costs, cost_multiplier=1.0)
        m = metrics_from_pnl(pnl, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
        if "error" in m:
            continue

        pos_d = _apply_delay(pos_seg, int(delay_bars))
        pnl_d, _ = pnl_from_positions(pos_d, ret_seg, costs, cost_multiplier=1.0)
        m_d = metrics_from_pnl(pnl_d, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
        if "error" in m_d:
            continue

        pnl_c, _ = pnl_from_positions(pos_seg, ret_seg, costs, cost_multiplier=float(stress_cost_mult))
        m_c = metrics_from_pnl(pnl_c, bar_seconds, mdd_floor=mdd_floor, calmar_cap=calmar_cap)
        if "error" in m_c:
            continue

        sh = float(m.get("sharpe", 0.0))
        sh_d = float(m_d.get("sharpe", 0.0))
        sh_c = float(m_c.get("sharpe", 0.0))

        rt = float(m.get("total_return", 0.0))
        rt_d = float(m_d.get("total_return", 0.0))
        rt_c = float(m_c.get("total_return", 0.0))

        fold_sh.append(sh)
        fold_sh_d1.append(sh_d)
        fold_sh_cost.append(sh_c)

        fold_rt.append(rt)
        fold_rt_d1.append(rt_d)
        fold_rt_cost.append(rt_c)

        fold_start_end.append((int(a), int(b)))

        if rt > 0:
            pos_folds += 1
        if rt_c > 0:
            pos_folds_cost += 1

    if len(fold_sh) < 2:
        return {"error": "cv_insufficient_folds", "folds_ok": int(len(fold_sh)), "k": int(k)}

    sh0 = np.asarray(fold_sh, dtype=np.float64)
    sh1 = np.asarray(fold_sh_d1, dtype=np.float64)
    shc = np.asarray(fold_sh_cost, dtype=np.float64)

    summary = {
        "n_folds": int(len(fold_sh)),
        "fold_start_end": fold_start_end,
        "positive_folds": int(pos_folds),
        "positive_folds_coststress": int(pos_folds_cost),

        "median_sharpe": float(np.median(sh0)),
        "min_sharpe": float(np.min(sh0)),
        "std_sharpe": float(np.std(sh0)) if sh0.size > 1 else 0.0,

        "median_sharpe_delay1": float(np.median(sh1)),
        "min_sharpe_delay1": float(np.min(sh1)),

        "median_sharpe_coststress": float(np.median(shc)),
        "min_sharpe_coststress": float(np.min(shc)),

        "fold_sharpes": [float(x) for x in fold_sh],
        "fold_sharpes_delay1": [float(x) for x in fold_sh_d1],
        "fold_sharpes_coststress": [float(x) for x in fold_sh_cost],

        "fold_total_returns": [float(x) for x in fold_rt],
        "fold_total_returns_delay1": [float(x) for x in fold_rt_d1],
        "fold_total_returns_coststress": [float(x) for x in fold_rt_cost],

        "warmup": int(warmup),
        "delay_bars": int(delay_bars),
        "stress_cost_mult": float(stress_cost_mult),
    }
    return summary


def _score_v4(
    m_train: dict[str, Any],
    m_hold: dict[str, Any],
    m_hold_stress: dict[str, Any],
    wf: dict[str, Any],
    trade_train: dict[str, Any],
    trade_hold: dict[str, Any],
    cv: dict[str, Any],
    regime: dict[str, Any] | None,
) -> float:
    """
    CV-first score + regime balance penalty.
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

    dominance_wf = abs(float(wf.get("dominance", 1.0)))
    turnover = float(trade_train.get("turnover_mean", 0.0))
    hold_mdd = abs(float(m_hold.get("max_drawdown", 0.0)))

    s = (
        0.55 * cv_med +
        0.22 * cv_min +
        0.10 * cv_med_d +
        0.08 * cv_med_c +
        0.05 * stress_sh +
        0.04 * hold_sh
    )
    s *= (0.85 + 0.15 * pos_frac)
    s -= 0.22 * cv_std
    s -= 0.22 * gap
    s -= 0.12 * hold_mdd
    s -= 0.05 * turnover
    s -= 0.12 * max(0.0, dominance_wf - 0.60)

    # regime adjustments
    if regime and "error" not in regime:
        cov = float(regime.get("coverage", 0.0)) / 9.0
        dom = float(regime.get("dominance", 0.0))
        worst = float(regime.get("worst_sharpe", 0.0))

        s += 0.10 * cov
        s -= 0.25 * max(0.0, dom - 0.55)
        s -= 0.08 * max(0.0, (-worst) - 0.20)

    return float(s)

def _soft_elite_score_stage1(
    m_train: dict[str, Any],
    m_hold: dict[str, Any],
    m_hold_stress: dict[str, Any] | None,
    wf: dict[str, Any] | None,
    trade_train: dict[str, Any],
    trade_hold: dict[str, Any],
) -> float:
    """
    Soft guidance score (NOT published). Used to keep evo search moving even when hard gates accept 0.
    Should correlate with OOS robustness but be cheap and not depend on CV.
    """
    train_sh = float(m_train.get("sharpe", 0.0))
    hold_sh = float(m_hold.get("sharpe", 0.0))

    if m_hold_stress is not None:
        stress_sh = float(m_hold_stress.get("sharpe", hold_sh))
    else:
        stress_sh = hold_sh

    gap = max(0.0, train_sh - hold_sh)
    hold_mdd = abs(float(m_hold.get("max_drawdown", 0.0)))

    turnover = float(trade_hold.get("turnover_mean", trade_train.get("turnover_mean", 0.0)))
    dom = abs(float(wf.get("dominance", 0.0))) if wf else 0.0

    s = 0.55 * hold_sh + 0.25 * train_sh + 0.10 * stress_sh
    s -= 0.20 * gap
    s -= 0.15 * hold_mdd
    s -= 0.05 * turnover
    s -= 0.10 * max(0.0, dom - 0.65)
    return float(s)


def evaluate_genomes_batch(genomes: list[dict[str, Any]], eval_cfg: EvalConfig) -> dict[str, Any]:
    """
    Patch 0006:
    - Returns accepted strategies AND (soft) elite_updates for non-stalling evo.
    - Returns reject reason counts for diagnostics.
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

    # Batch-level guards (still return diagnostics)
    if train_ret.size < int(eval_cfg.min_train_bars):
        return {
            "accepted": [],
            "elite_updates": [],
            "rejects": {"batch_train_too_short": int(len(genomes))},
        }
    if holdout_ret.size < int(eval_cfg.min_holdout_bars):
        return {
            "accepted": [],
            "elite_updates": [],
            "rejects": {"batch_holdout_too_short": int(len(genomes))},
        }

    # Regime cache (lazy)
    regime_arr = None
    regime_info = None
    if bool(eval_cfg.regime_enabled):
        rc = _get_or_build_regime_cache(int(eval_cfg.regime_vol_window), int(eval_cfg.regime_trend_window))
        regime_arr = rc.get("regime", None)
        regime_info = (rc.get("info", {}) or {})

    rejects = Counter()
    out: list[dict[str, Any]] = []
    elite_scores: list[tuple[float, dict[str, Any]]] = []

    def _rej(k: str) -> None:
        rejects[k] += 1

    def _add_elite(soft: float, genome: dict[str, Any]) -> None:
        if not math.isfinite(float(soft)):
            return
        elite_scores.append((float(soft), genome))

    for genome in genomes:
        try:
            if genome.get("type") != "linear_alpha_v1":
                _rej("fail_type")
                continue

            # Build alpha
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

            # -------------------------
            # Stage 1: Train
            # -------------------------
            pos_train = pos[: split_close + 1]
            pnl_train, trade_train = pnl_from_positions(pos_train, train_ret, eval_cfg.costs, cost_multiplier=1.0)
            m_train = metrics_from_pnl(pnl_train, bar_seconds, mdd_floor=eval_cfg.mdd_floor, calmar_cap=eval_cfg.calmar_cap)
            if "error" in m_train:
                _rej("fail_train_metrics_error")
                continue

            if trade_train["exposure"] < eval_cfg.min_exposure_train:
                _rej("fail_train_exposure")
                continue
            if trade_train["active_bars"] < eval_cfg.min_active_bars_train:
                _rej("fail_train_active_bars")
                continue
            if trade_train["trades"] < eval_cfg.min_trades_train:
                _rej("fail_train_trades")
                continue
            if trade_train["turnover_mean"] > eval_cfg.max_turnover_mean_train:
                _rej("fail_train_turnover")
                continue
            if abs(m_train["max_drawdown"]) > eval_cfg.max_drawdown_abs_train:
                _rej("fail_train_mdd")
                continue
            if m_train["sharpe"] < eval_cfg.min_train_sharpe:
                _rej("fail_train_sharpe")
                continue

            wf = walkforward_metrics(
                pnl_train,
                bar_seconds,
                n_splits=eval_cfg.n_walkforward_splits,
                mdd_floor=eval_cfg.mdd_floor,
                calmar_cap=eval_cfg.calmar_cap,
            )
            if "error" in wf:
                _rej("fail_train_wf_error")
                continue
            if wf["positive_segments"] < eval_cfg.min_wf_positive_segments:
                _rej("fail_train_wf_pos_segments")
                continue
            if abs(wf["dominance"]) > eval_cfg.max_wf_dominance:
                _rej("fail_train_wf_dominance")
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

            # -------------------------
            # Stage 1: Holdout (OOS)
            # -------------------------
            pos_hold = pos[split_close:]
            pnl_hold, trade_hold = pnl_from_positions(pos_hold, holdout_ret, eval_cfg.costs, cost_multiplier=1.0)
            m_hold = metrics_from_pnl(pnl_hold, bar_seconds, mdd_floor=eval_cfg.mdd_floor, calmar_cap=eval_cfg.calmar_cap)
            if "error" in m_hold:
                _rej("fail_holdout_metrics_error")
                continue

            if trade_hold["exposure"] < eval_cfg.min_exposure_holdout:
                _rej("fail_holdout_exposure")
                continue
            if trade_hold["active_bars"] < eval_cfg.min_active_bars_holdout:
                _rej("fail_holdout_active_bars")
                continue
            if trade_hold["trades"] < eval_cfg.min_trades_holdout:
                _rej("fail_holdout_trades")
                continue
            if trade_hold["turnover_mean"] > eval_cfg.max_turnover_mean_holdout:
                _rej("fail_holdout_turnover")
                continue
            if abs(m_hold["max_drawdown"]) > eval_cfg.max_drawdown_abs_holdout:
                _rej("fail_holdout_mdd")
                continue

            # soft score available after we have train+holdout (no stress yet)
            soft_stage1 = _soft_elite_score_stage1(
                m_train=m_train,
                m_hold=m_hold,
                m_hold_stress=None,
                wf=wf,
                trade_train=trade_train,
                trade_hold=trade_hold,
            )

            if m_hold["sharpe"] < eval_cfg.min_holdout_sharpe:
                _add_elite(soft_stage1, genome)
                _rej("fail_holdout_sharpe")
                continue

            pnl_hold_s, _ = pnl_from_positions(
                pos_hold, holdout_ret, eval_cfg.costs, cost_multiplier=float(eval_cfg.stress_cost_mult)
            )
            m_hold_stress = metrics_from_pnl(
                pnl_hold_s, bar_seconds, mdd_floor=eval_cfg.mdd_floor, calmar_cap=eval_cfg.calmar_cap
            )
            if "error" in m_hold_stress:
                _add_elite(soft_stage1, genome)
                _rej("fail_holdout_stress_metrics_error")
                continue

            soft_stage1_stress = _soft_elite_score_stage1(
                m_train=m_train,
                m_hold=m_hold,
                m_hold_stress=m_hold_stress,
                wf=wf,
                trade_train=trade_train,
                trade_hold=trade_hold,
            )

            if m_hold_stress["sharpe"] < eval_cfg.min_holdout_stress_sharpe:
                _add_elite(soft_stage1_stress, genome)
                _rej("fail_holdout_stress_sharpe")
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

            # -------------------------
            # Stage 2: CV (Deep Validation)
            # -------------------------
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
                    _add_elite(soft_stage1_stress, genome)
                    _rej(f"fail_cv_{str(cv_summary.get('error'))}")
                    continue

                n_folds_ok = int(cv_summary.get("n_folds", 0))
                pos_folds = int(cv_summary.get("positive_folds", 0))

                # positive folds (base)
                min_pos = int(eval_cfg.cv_min_positive_folds)
                if min_pos <= 0:
                    min_pos = int(math.ceil(float(eval_cfg.cv_min_positive_folds_frac) * float(n_folds_ok)))
                min_pos = max(1, min(min_pos, n_folds_ok))
                if pos_folds < min_pos:
                    _add_elite(soft_stage1_stress, genome)
                    _rej("fail_cv_pos_folds")
                    continue

                # base gates
                if float(cv_summary.get("min_sharpe", -1e9)) < float(eval_cfg.cv_min_sharpe):
                    _add_elite(soft_stage1_stress, genome)
                    _rej("fail_cv_min_sharpe")
                    continue
                if float(cv_summary.get("median_sharpe", -1e9)) < float(eval_cfg.cv_min_median_sharpe_base):
                    _add_elite(soft_stage1_stress, genome)
                    _rej("fail_cv_med_sharpe")
                    continue
                if float(cv_summary.get("median_sharpe_delay1", -1e9)) < float(eval_cfg.cv_min_median_sharpe_delay1):
                    _add_elite(soft_stage1_stress, genome)
                    _rej("fail_cv_med_delay1")
                    continue
                if float(cv_summary.get("median_sharpe_coststress", -1e9)) < float(eval_cfg.cv_min_median_sharpe_coststress):
                    _add_elite(soft_stage1_stress, genome)
                    _rej("fail_cv_med_cost")
                    continue

                # v2 gates
                if float(cv_summary.get("min_sharpe_delay1", -1e9)) < float(eval_cfg.cv_min_sharpe_delay1):
                    _add_elite(soft_stage1_stress, genome)
                    _rej("fail_cv_min_delay1")
                    continue
                if float(cv_summary.get("min_sharpe_coststress", -1e9)) < float(eval_cfg.cv_min_sharpe_coststress):
                    _add_elite(soft_stage1_stress, genome)
                    _rej("fail_cv_min_cost")
                    continue

                pos_folds_cost = int(cv_summary.get("positive_folds_coststress", 0))
                min_pos_cost = int(eval_cfg.cv_min_positive_folds_coststress)
                if min_pos_cost <= 0:
                    min_pos_cost = int(
                        math.ceil(float(eval_cfg.cv_min_positive_folds_coststress_frac) * float(n_folds_ok))
                    )
                min_pos_cost = max(1, min(min_pos_cost, n_folds_ok))
                if pos_folds_cost < min_pos_cost:
                    _add_elite(soft_stage1_stress, genome)
                    _rej("fail_cv_pos_cost_folds")
                    continue

                metrics["cv"] = cv_summary
                pass_flags = "train+wf+holdout+stress+cv"

            # -------------------------
            # Regime attribution (non-gating)
            # -------------------------
            regime_metrics = None
            if bool(eval_cfg.regime_enabled) and regime_arr is not None:
                try:
                    pnl_full = np.concatenate([pnl_train, pnl_hold])
                    regime_metrics = _regime_attribution(
                        pos=pos,
                        pnl_full=pnl_full,
                        regime=regime_arr,
                        bar_seconds=bar_seconds,
                        min_bars=int(eval_cfg.regime_min_bars),
                        min_exposure=float(eval_cfg.regime_min_exposure),
                    )
                    if regime_info:
                        regime_metrics = dict(regime_metrics)
                        regime_metrics["info"] = regime_info
                    metrics["regime"] = regime_metrics
                    pass_flags += "+regime"
                except Exception:
                    _rej("warn_regime_exception")

            # -------------------------
            # Score
            # -------------------------
            if cv_summary is not None and "error" not in cv_summary:
                score = _score_v4(
                    m_train, m_hold, m_hold_stress, wf, trade_train, trade_hold, cv_summary, regime_metrics
                )
            else:
                # fallback (deep_validation disabled)
                score = 0.55 * float(m_hold.get("sharpe", 0.0)) + 0.25 * float(m_train.get("sharpe", 0.0))

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
            _rej("fail_exception")
            continue

    # Choose a few best near-miss candidates as soft elite updates
    elite_updates: list[dict[str, Any]] = []
    if elite_scores:
        elite_scores.sort(key=lambda x: x[0], reverse=True)
        k = int(max(2, min(8, math.ceil(0.06 * max(1, len(genomes))))))
        for sc, g in elite_scores[:k]:
            elite_updates.append({"score": float(sc), "genome": g})

    return {
        "accepted": out,
        "elite_updates": elite_updates,
        "rejects": dict(rejects),
    }


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
    best_published_score = -1e9

    t0 = time.time()
    last_report = t0

    # recent diagnostics window (since last on_progress)
    reject_total = Counter()
    reject_recent = Counter()
    elite_updates_recent = 0

    ctx = mp.get_context("spawn")

    cand = CandidateGenerator(
        rng=rng,
        space=cfg.search_space,
        mode=str(cfg.search_mode),
        elite_size=int(cfg.elite_size),
        seen_max=int(cfg.seen_max),
        frac_random=float(cfg.evo_frac_random),
        frac_mutate=float(cfg.evo_frac_mutate),
        frac_crossover=float(cfg.evo_frac_crossover),
        seed_genomes=list(cfg.elite_seed_genomes),
    )

    # seed elite pool (low-score placeholders) so mutate/crossover works immediately
    for g in cfg.elite_seed_genomes[: min(128, len(cfg.elite_seed_genomes))]:
        cand.update_elite(score=0.0, genome=g)

    def _consume_batch_result(batch_res: Any) -> None:
        nonlocal total_accepted, best_published_score, elite_updates_recent

        accepted: list[dict[str, Any]] = []
        elite_updates: list[dict[str, Any]] = []
        rejects: dict[str, Any] = {}

        if isinstance(batch_res, dict) and "accepted" in batch_res:
            accepted = list(batch_res.get("accepted") or [])
            elite_updates = list(batch_res.get("elite_updates") or [])
            rejects = dict(batch_res.get("rejects") or {})
        else:
            # Backward compatibility (if something returns a list)
            accepted = list(batch_res or [])

        # rejects accounting
        if rejects:
            for k, v in rejects.items():
                try:
                    n = int(v)
                except Exception:
                    continue
                if n <= 0:
                    continue
                reject_total[k] += n
                reject_recent[k] += n

        # soft elite updates (non-published)
        if elite_updates:
            elite_updates_recent += int(len(elite_updates))
            for u in elite_updates:
                try:
                    g = u.get("genome", None)
                    if not isinstance(g, dict):
                        continue
                    sc = float(u.get("score", -1e9))
                    cand.update_elite(sc, g)
                except Exception:
                    continue

        # accepted / published
        for r in accepted:
            total_accepted += 1
            sc = float(r.get("score", -1e9))
            if sc > best_published_score:
                best_published_score = sc
            try:
                cand.update_elite(sc, r["genome"])
            except Exception:
                pass
            on_result(r)

    with ProcessPoolExecutor(
        max_workers=int(cfg.workers),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(cfg.ts_path, cfg.ohlcv_path, int(cfg.bar_seconds)),
    ) as ex:
        pending = set()
        max_in_flight = int(max(1, cfg.max_in_flight))

        while time.time() < deadline:
            # submit more work
            while len(pending) < max_in_flight and time.time() < deadline:
                genomes = cand.next_batch(cfg.batch_size)
                if not genomes:
                    break
                fut = ex.submit(evaluate_genomes_batch, genomes, cfg.eval_cfg)
                pending.add(fut)
                total_tested += len(genomes)

            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

            for fut in done:
                try:
                    batch_res = fut.result()
                except Exception:
                    # we don't know exact reject reason here
                    continue
                _consume_batch_result(batch_res)

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
                        # published best (db-worthy)
                        "best_score": best_published_score,
                        # elite best (guidance; includes soft updates)
                        "elite_best": cand.elite.best_score(),
                        "elite_size": cand.elite.size(),
                        "seen": len(cand.seen),
                        "in_flight": len(pending),
                        "elite_updates_recent": int(elite_updates_recent),
                        "reject_counts_recent": dict(reject_recent),
                        "rejects_recent_total": int(sum(reject_recent.values())),
                        "reject_top": list(reject_recent.most_common(6)),
                        "rejects_total": int(sum(reject_total.values())),
                    }
                )
                # reset recent window
                reject_recent.clear()
                elite_updates_recent = 0
                last_report = now

        # drain remaining
        if pending:
            done, _ = wait(pending, timeout=5.0)
            for fut in done:
                try:
                    batch_res = fut.result()
                except Exception:
                    continue
                _consume_batch_result(batch_res)

    dt = max(1e-9, time.time() - t0)
    return {
        "tested": total_tested,
        "accepted": total_accepted,
        "tested_per_sec": total_tested / dt,
        "accepted_pct": (total_accepted / max(1, total_tested)) * 100.0,
        "best_score": best_published_score,
        "elite_best": cand.elite.best_score(),
        "seconds": dt,
        "rejects_total": int(sum(reject_total.values())),
        "reject_top": list(reject_total.most_common(10)),
    }
