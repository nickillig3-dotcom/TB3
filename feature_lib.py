#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature library for Strategy‑Miner (Patch 0002).

Features are computed lazily and cached per worker process.
All computations are vectorized with NumPy.

Naming convention:
- mom_ir: momentum info-ratio (rolling mean(ret)/rolling std(ret))
- mr_z: mean reversion z-score: -(close - ma)/std(close)
- vol_s: volatility scaled: (rolling std(ret) / median_vol) - 1
- volu_z: volume z-score: (vol - mean(vol))/std(vol)

All features return float32 arrays length N (aligned to close array length).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import numpy as np


EPS = 1e-12


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    w = int(window)
    if w <= 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / float(w)
    # pad to length
    pad = np.full(w - 1, out[0], dtype=np.float64) if out.size > 0 else np.zeros(w - 1, dtype=np.float64)
    return np.concatenate([pad, out])


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    w = int(window)
    if w <= 1:
        return np.full_like(x, 0.0, dtype=np.float64)
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    sum1 = c1[w:] - c1[:-w]
    sum2 = c2[w:] - c2[:-w]
    mean = sum1 / float(w)
    var = (sum2 / float(w)) - (mean * mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    pad = np.full(w - 1, std[0], dtype=np.float64) if std.size > 0 else np.zeros(w - 1, dtype=np.float64)
    return np.concatenate([pad, std])


def _clip_z(x: np.ndarray, clip: float = 6.0) -> np.ndarray:
    return np.clip(x, -clip, clip)


@dataclass
class FeatureStore:
    close: np.ndarray
    volume: np.ndarray | None
    simple_ret: np.ndarray  # length N-1
    max_cache_items: int = 24

    def __post_init__(self) -> None:
        # convert to float64 view for computations
        self._close_f64 = self.close.astype(np.float64, copy=False)
        self._ret_f64 = self.simple_ret.astype(np.float64, copy=False)
        self._cache: "OrderedDict[Tuple[str,int], np.ndarray]" = OrderedDict()
        self._scalar_cache: dict[Tuple[str,int], float] = {}

    def _cache_put(self, key: Tuple[str, int], arr: np.ndarray) -> np.ndarray:
        self._cache[key] = arr
        self._cache.move_to_end(key)
        while len(self._cache) > int(self.max_cache_items):
            self._cache.popitem(last=False)
        return arr

    def get(self, name: str, window: int) -> np.ndarray:
        key = (str(name), int(window))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if name == "mom_ir":
            arr = self._mom_ir(window)
        elif name == "mr_z":
            arr = self._mr_z(window)
        elif name == "vol_s":
            arr = self._vol_s(window)
        elif name == "volu_z":
            arr = self._volu_z(window)
        else:
            raise KeyError(f"Unknown feature: {name}")

        return self._cache_put(key, arr)

    def _mom_ir(self, window: int) -> np.ndarray:
        # rolling mean(ret)/rolling std(ret); pad to length N by prepending 0
        w = int(window)
        r = self._ret_f64
        mu = _rolling_mean(r, w)
        sig = _rolling_std(r, w) + EPS
        ir = mu / sig
        ir = _clip_z(ir)
        # align to close length N: ret has N-1; prepend first value
        out = np.concatenate([[ir[0]], ir]).astype(np.float32)
        return out

    def _mr_z(self, window: int) -> np.ndarray:
        w = int(window)
        c = self._close_f64
        ma = _rolling_mean(c, w)
        sd = _rolling_std(c, w) + EPS
        z = -(c - ma) / sd
        z = _clip_z(z)
        return z.astype(np.float32)

    def _vol_s(self, window: int) -> np.ndarray:
        w = int(window)
        r = self._ret_f64
        vol = _rolling_std(r, w)
        # scale by median vol to make magnitudes comparable
        med_key = ("vol_med", w)
        if med_key in self._scalar_cache:
            med = self._scalar_cache[med_key]
        else:
            med = float(np.median(vol[vol > 0])) if np.any(vol > 0) else float(np.median(vol))
            med = med if med > EPS else 1.0
            self._scalar_cache[med_key] = med

        vs = (vol / med) - 1.0
        vs = _clip_z(vs, clip=8.0)
        out = np.concatenate([[vs[0]], vs]).astype(np.float32)
        return out

    def _volu_z(self, window: int) -> np.ndarray:
        if self.volume is None:
            # If volume missing, provide zeros -> feature effectively unused
            return np.zeros_like(self.close, dtype=np.float32)
        w = int(window)
        v = self.volume.astype(np.float64, copy=False)
        mu = _rolling_mean(v, w)
        sd = _rolling_std(v, w) + EPS
        z = (v - mu) / sd
        z = _clip_z(z)
        return z.astype(np.float32)

    def vol_gate_mask(self, window: int, mode: str) -> np.ndarray:
        """
        Simple regime gate:
          - 'high': allow when vol_s(window) > 0
          - 'low' : allow when vol_s(window) < 0
          - 'any' : allow always
        """
        mode = (mode or "any").lower()
        if mode == "any":
            return np.ones_like(self.close, dtype=bool)

        vol_s = self.get("vol_s", int(window))
        if mode == "high":
            return vol_s > 0.0
        if mode == "low":
            return vol_s < 0.0

        return np.ones_like(self.close, dtype=bool)
