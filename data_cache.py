#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset ingestion + memmap cache builder (Patch 0002).

We store:
- cache/<dataset_id>_ts.npy    int64 timestamps (seconds/ms/ns ok)
- cache/<dataset_id>_ohlcv.npy float32 [N,5] columns: open, high, low, close, volume (volume may be 0 if missing)
- cache/<dataset_id>_meta.json metadata

The memmap cache is built once; workers load using np.load(..., mmap_mode='r').
"""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from sm_utils import ensure_dir, atomic_write_json


@dataclass(frozen=True)
class DatasetCache:
    dataset_id: str
    ts_path: str
    ohlcv_path: str
    meta_path: str
    meta: dict[str, Any]


def _fingerprint_path(path: str) -> str:
    """
    Cheap stable fingerprint based on absolute path + stat.
    (Avoids hashing full file contents.)
    """
    p = os.path.abspath(path)
    st = os.stat(p)
    s = f"{p}|{st.st_size}|{int(st.st_mtime_ns)}".encode("utf-8")
    import hashlib
    return hashlib.sha256(s).hexdigest()[:16]


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def generate_demo_ohlcv(
    n_bars: int = 120_000,
    *,
    seed: int = 7,
    start_ts: int | None = None,
    bar_seconds: int = 60 * 15,  # 15 min
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Generates a regime-switching synthetic market (not meant to be realistic,
    but useful to validate pipelines and for quick dev).
    """
    rng = np.random.default_rng(int(seed))
    n = int(n_bars)
    if start_ts is None:
        start_ts = int(time.time()) - n * int(bar_seconds)

    # Build piecewise regimes
    regimes = [
        # (drift, vol)
        (0.0, 0.004),
        (0.00005, 0.0025),
        (-0.00003, 0.0035),
        (0.00002, 0.006),
    ]
    n_reg = len(regimes)
    seg = max(1000, n // n_reg)
    drifts = np.zeros(n, dtype=np.float64)
    vols = np.zeros(n, dtype=np.float64)
    for i, (mu, sig) in enumerate(regimes):
        a = i * seg
        b = (i + 1) * seg if i < n_reg - 1 else n
        drifts[a:b] = mu
        vols[a:b] = sig

    # Log returns
    eps = rng.standard_normal(n) * vols + drifts
    logp = np.cumsum(eps) + np.log(100.0)
    close = np.exp(logp)

    # OHLC around close
    # open is prev close, high/low are close +/- noise
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.standard_normal(n) * (vols * close * 2.5))
    high = np.maximum.reduce([open_, close, close + spread])
    low = np.minimum.reduce([open_, close, close - spread])

    # Volume correlated with volatility
    volu = (1e3 * (1.0 + 80.0 * vols) * np.exp(rng.standard_normal(n) * 0.15)).astype(np.float64)

    ts = (start_ts + np.arange(n, dtype=np.int64) * int(bar_seconds)).astype(np.int64)

    ohlcv = np.stack([open_, high, low, close, volu], axis=1).astype(np.float32)

    meta = {
        "source": "demo",
        "created_utc": _now_iso(),
        "n_bars": int(n),
        "bar_seconds": int(bar_seconds),
        "columns": ["open", "high", "low", "close", "volume"],
        "platform": platform.platform(),
    }
    return ts, ohlcv, meta


def _coerce_float(x: str) -> float | None:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        return float(s)
    except Exception:
        return None


def _coerce_int(x: str) -> int | None:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        return int(float(s))
    except Exception:
        return None


def load_csv_ohlcv(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Loads a CSV with header columns.
    Requires at least 'close'. Accepts flexible names:
      ts: timestamp, time, date
      open/high/low/close: open, high, low, close
      volume: volume, vol
    """
    p = os.path.abspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)

    # Try pandas if available (faster, more flexible)
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        def pick(*names: str) -> str | None:
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_ts = pick("timestamp", "time", "date", "datetime")
        c_o = pick("open")
        c_h = pick("high")
        c_l = pick("low")
        c_c = pick("close", "adjclose", "adj_close")
        c_v = pick("volume", "vol")

        if c_c is None:
            raise ValueError("CSV must contain a 'close' column")

        close = df[c_c].astype("float64").to_numpy()
        open_ = df[c_o].astype("float64").to_numpy() if c_o else np.concatenate([[close[0]], close[:-1]])
        high = df[c_h].astype("float64").to_numpy() if c_h else np.maximum(open_, close)
        low = df[c_l].astype("float64").to_numpy() if c_l else np.minimum(open_, close)
        volu = df[c_v].astype("float64").to_numpy() if c_v else np.zeros_like(close)

        if c_ts:
            ts_raw = df[c_ts].to_numpy()
            # best-effort: if already numeric, keep; else parse to int seconds
            if np.issubdtype(ts_raw.dtype, np.number):
                ts = ts_raw.astype(np.int64)
            else:
                ts = pd.to_datetime(df[c_ts], errors="coerce").astype("int64")  # ns
                ts = ts.to_numpy().astype(np.int64)
        else:
            ts = np.arange(close.shape[0], dtype=np.int64)

        # Drop NaNs
        mask = np.isfinite(close) & np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(volu)
        close = close[mask]; open_ = open_[mask]; high = high[mask]; low = low[mask]; volu = volu[mask]; ts = ts[mask]

        # Sort by ts
        idx = np.argsort(ts)
        ts = ts[idx]
        ohlcv = np.stack([open_[idx], high[idx], low[idx], close[idx], volu[idx]], axis=1).astype(np.float32)

        meta = {
            "source": "csv",
            "path": p,
            "created_utc": _now_iso(),
            "n_bars": int(ts.shape[0]),
            "columns": ["open", "high", "low", "close", "volume"],
            "ts_column": c_ts,
        }
        return ts.astype(np.int64), ohlcv, meta
    except ImportError:
        pass

    # Fallback: csv.DictReader
    with open(p, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        cols = {c.lower(): c for c in reader.fieldnames}

        def pick(*names: str) -> str | None:
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_ts = pick("timestamp", "time", "date", "datetime")
        c_o = pick("open")
        c_h = pick("high")
        c_l = pick("low")
        c_c = pick("close", "adjclose", "adj_close")
        c_v = pick("volume", "vol")

        if c_c is None:
            raise ValueError("CSV must contain a 'close' column")

        ts_list = []
        o_list = []
        h_list = []
        l_list = []
        c_list = []
        v_list = []

        for row in reader:
            c = _coerce_float(row.get(c_c))
            if c is None:
                continue
            o = _coerce_float(row.get(c_o)) if c_o else c
            h = _coerce_float(row.get(c_h)) if c_h else max(o, c) if o is not None else c
            l = _coerce_float(row.get(c_l)) if c_l else min(o, c) if o is not None else c
            v = _coerce_float(row.get(c_v)) if c_v else 0.0

            if o is None or h is None or l is None or v is None:
                continue

            if c_ts:
                t = _coerce_int(row.get(c_ts))
                if t is None:
                    continue
            else:
                t = len(ts_list)

            ts_list.append(int(t))
            o_list.append(float(o))
            h_list.append(float(h))
            l_list.append(float(l))
            c_list.append(float(c))
            v_list.append(float(v))

    ts = np.asarray(ts_list, dtype=np.int64)
    o = np.asarray(o_list, dtype=np.float64)
    h = np.asarray(h_list, dtype=np.float64)
    l = np.asarray(l_list, dtype=np.float64)
    c = np.asarray(c_list, dtype=np.float64)
    v = np.asarray(v_list, dtype=np.float64)

    idx = np.argsort(ts)
    ts = ts[idx]
    ohlcv = np.stack([o[idx], h[idx], l[idx], c[idx], v[idx]], axis=1).astype(np.float32)

    meta = {
        "source": "csv",
        "path": p,
        "created_utc": _now_iso(),
        "n_bars": int(ts.shape[0]),
        "columns": ["open", "high", "low", "close", "volume"],
        "ts_column": c_ts,
    }
    return ts, ohlcv, meta


def build_or_load_cache_from_arrays(
    dataset_id: str,
    ts: np.ndarray,
    ohlcv: np.ndarray,
    meta: dict[str, Any],
    *,
    cache_dir: str = "cache",
    force_rebuild: bool = False,
) -> DatasetCache:
    ensure_dir(cache_dir)
    ts_path = os.path.join(cache_dir, f"{dataset_id}_ts.npy")
    ohlcv_path = os.path.join(cache_dir, f"{dataset_id}_ohlcv.npy")
    meta_path = os.path.join(cache_dir, f"{dataset_id}_meta.json")

    if (not force_rebuild) and os.path.exists(ts_path) and os.path.exists(ohlcv_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_loaded = json.load(f)
        return DatasetCache(dataset_id, ts_path, ohlcv_path, meta_path, meta_loaded)

    # Write arrays
    np.save(ts_path, ts.astype(np.int64))
    np.save(ohlcv_path, ohlcv.astype(np.float32))
    atomic_write_json(meta_path, meta)

    return DatasetCache(dataset_id, ts_path, ohlcv_path, meta_path, meta)


def prepare_demo_dataset(
    *,
    n_bars: int = 120_000,
    seed: int = 7,
    bar_seconds: int = 60 * 15,
    cache_dir: str = "cache",
    force_rebuild: bool = False,
) -> DatasetCache:
    ts, ohlcv, meta = generate_demo_ohlcv(n_bars=n_bars, seed=seed, bar_seconds=bar_seconds)
    dataset_id = f"demo_{n_bars}_{bar_seconds}_{seed}"
    return build_or_load_cache_from_arrays(
        dataset_id=dataset_id,
        ts=ts,
        ohlcv=ohlcv,
        meta=meta,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )


def prepare_csv_dataset(
    csv_path: str,
    *,
    cache_dir: str = "cache",
    force_rebuild: bool = False,
) -> DatasetCache:
    ts, ohlcv, meta = load_csv_ohlcv(csv_path)
    dataset_id = f"csv_{_fingerprint_path(csv_path)}"
    return build_or_load_cache_from_arrays(
        dataset_id=dataset_id,
        ts=ts,
        ohlcv=ohlcv,
        meta=meta,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )


def load_memmap(cache: DatasetCache) -> tuple[np.ndarray, np.ndarray]:
    ts = np.load(cache.ts_path, mmap_mode="r")
    ohlcv = np.load(cache.ohlcv_path, mmap_mode="r")
    return ts, ohlcv
