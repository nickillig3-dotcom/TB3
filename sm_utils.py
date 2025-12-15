#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for Strategy‑Miner (Patch 0002).

Design goals:
- Zero external dependencies.
- Windows-safe multiprocessing (spawn).
- Deterministic hashing for strategy genomes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import sys
import uuid
from typing import Any


def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def local_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def short_run_id() -> str:
    return uuid.uuid4().hex[:10]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_thread_env(threads: int = 1) -> None:
    """
    Prevent BLAS/OpenMP oversubscription when we also use multiprocessing.
    Call this BEFORE creating the process pool (ideally early in main).
    """
    threads = int(max(1, threads))
    env = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any) -> str:
    """
    Stable sha256 hash over json serialization (sorted keys).
    """
    s = stable_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def atomic_write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def setup_logger(name: str, latest_path: str, run_path: str) -> logging.Logger:
    """
    File logger + console logger.
    """
    ensure_dir(os.path.dirname(os.path.abspath(latest_path)) or ".")
    ensure_dir(os.path.dirname(os.path.abspath(run_path)) or ".")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Latest (overwrite)
    fh_latest = logging.FileHandler(latest_path, mode="w", encoding="utf-8")
    fh_latest.setLevel(logging.INFO)
    fh_latest.setFormatter(fmt)
    logger.addHandler(fh_latest)

    # Run log (unique)
    fh_run = logging.FileHandler(run_path, mode="w", encoding="utf-8")
    fh_run.setLevel(logging.INFO)
    fh_run.setFormatter(fmt)
    logger.addHandler(fh_run)

    return logger


def read_json_if_exists(path: str) -> dict | None:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def get_recommended_workers_from_capabilities(path: str = "capabilities_latest.json") -> int | None:
    """
    Best-effort: returns recommended_cpu_workers if present.
    """
    cap = read_json_if_exists(path)
    if not cap:
        return None
    try:
        rec = cap.get("recommendations", {})
        w = rec.get("recommended_cpu_workers")
        if w is None:
            return None
        w = int(w)
        if w >= 1:
            return w
    except Exception:
        return None
    return None
