"""
resource_manager.py

Strategy‑Miner — auto resource tuning + capabilities loading.

Patch 0007:
- liest machine capabilities aus capabilities_latest.json (falls vorhanden)
- empfiehlt workers/batch/inflight, wenn CLI-Werte -1 (=auto) sind
- ist dependency-free (nur stdlib)
- darf Strategy‑Miner NIEMALS crashen -> bei Fehlern: safe defaults
"""
from __future__ import annotations

import json
import os
import sys
import shutil
from typing import Any, Dict, Optional


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp_int(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def load_capabilities_latest(path: str = "capabilities_latest.json") -> Dict[str, Any]:
    """Load capabilities_latest.json if present; otherwise return {}."""
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _total_ram_bytes() -> int:
    """Best-effort total RAM bytes; returns 0 on failure."""
    # Try psutil if available (not required).
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    # Windows via GlobalMemoryStatusEx
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return 0
            return int(stat.ullTotalPhys)
        except Exception:
            return 0

    # POSIX via sysconf
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[arg-type]
            page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[arg-type]
            if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
                return int(pages * page_size)
    except Exception:
        pass

    return 0


def total_ram_gb() -> float:
    b = _total_ram_bytes()
    if b <= 0:
        return 0.0
    return b / (1024.0 ** 3)


def free_disk_gb(path: str = ".") -> float:
    try:
        usage = shutil.disk_usage(path)
        return float(usage.free) / (1024.0 ** 3)
    except Exception:
        return 0.0


def recommend_resources(args: Any, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Recommend {'workers','batch','inflight','reason'}.

    Patch 0007 convention:
      - args.workers / args.batch / args.inflight:
          -1 means "auto"
           >0 means "explicit"
    """
    caps = capabilities or {}
    logical = os.cpu_count() or 4

    # If capabilities file provides a better core estimate, use it (flexible keys).
    phys = 0
    for k in ("cpu_physical_cores", "physical_cores", "cores_physical", "cpu_cores_physical"):
        phys = max(phys, _safe_int(caps.get(k, 0), 0))
    core_basis = phys if phys > 0 else logical

    # Auto workers: leave 1 core for OS/UI if possible.
    auto_workers = max(1, min(logical, core_basis))
    if auto_workers >= 4:
        auto_workers = max(1, auto_workers - 1)

    # Respect explicit user values.
    req_workers = _safe_int(getattr(args, "workers", -1), -1)
    req_batch = _safe_int(getattr(args, "batch", -1), -1)
    req_inflight = _safe_int(getattr(args, "inflight", -1), -1)

    workers = req_workers if req_workers > 0 else auto_workers

    # Match your observed good defaults: batch≈12×workers, inflight≈2×workers
    auto_inflight = _clamp_int(2 * workers, lo=max(4, workers), hi=max(16, 8 * workers))
    inflight = req_inflight if req_inflight > 0 else auto_inflight

    auto_batch = _clamp_int(12 * workers, lo=32, hi=4096)
    batch = req_batch if req_batch > 0 else auto_batch

    ram_gb = total_ram_gb()
    disk_free_gb = free_disk_gb(".")

    reason = (
        f"logical_cores={logical}, physical_cores={phys or 'n/a'}, "
        f"ram_gb={ram_gb:.1f}, disk_free_gb={disk_free_gb:.1f}, "
        f"auto_workers={auto_workers}, auto_batch={auto_batch}, auto_inflight={auto_inflight}"
    )

    return {"workers": int(workers), "batch": int(batch), "inflight": int(inflight), "reason": reason}
