#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner — Patch 0001
System Capability Probe + Micro‑Benchmarks

Outputs:
  logs/benchmark_latest.log
  logs/benchmark_<timestamp>_<runid>.log
  capabilities_latest.json
  capabilities_<timestamp>_<runid>.json

No external dependencies required (optional: psutil, numpy).
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------
# Utilities
# ----------------------------

def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def local_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def short_run_id() -> str:
    return uuid.uuid4().hex[:8]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def human_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.2f} {units[i]}"


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def atomic_write_text(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    ensure_dir(d)
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, obj: dict) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


def setup_logger(run_ts: str, run_id: str) -> tuple[logging.Logger, str, str]:
    ensure_dir("logs")
    latest_log = os.path.join("logs", "benchmark_latest.log")
    run_log = os.path.join("logs", f"benchmark_{run_ts}_{run_id}.log")

    logger = logging.getLogger("system_benchmark")
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

    # latest log (overwrite)
    fh_latest = logging.FileHandler(latest_log, mode="w", encoding="utf-8")
    fh_latest.setLevel(logging.INFO)
    fh_latest.setFormatter(fmt)
    logger.addHandler(fh_latest)

    # run log (unique)
    fh_run = logging.FileHandler(run_log, mode="w", encoding="utf-8")
    fh_run.setLevel(logging.INFO)
    fh_run.setFormatter(fmt)
    logger.addHandler(fh_run)

    return logger, latest_log, run_log


# ----------------------------
# System introspection
# ----------------------------

def try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def try_import_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except Exception:
        return None


def get_cpu_counts(psutil_mod) -> tuple[int | None, int | None]:
    logical = os.cpu_count()
    physical = None

    if psutil_mod is not None:
        try:
            physical = psutil_mod.cpu_count(logical=False)
        except Exception:
            physical = None

    # Fallback attempts if psutil missing
    if physical is None:
        try:
            if sys.platform.startswith("linux"):
                physical = _linux_physical_cores()
            elif sys.platform == "darwin":
                physical = _macos_physical_cores()
            elif os.name == "nt":
                physical = _windows_physical_cores()
        except Exception:
            physical = None

    return logical, physical


def _linux_physical_cores() -> int | None:
    # Count unique (physical id, core id) pairs from /proc/cpuinfo
    path = "/proc/cpuinfo"
    if not os.path.exists(path):
        return None
    phys_id = None
    core_id = None
    pairs = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                if phys_id is not None and core_id is not None:
                    pairs.add((phys_id, core_id))
                phys_id, core_id = None, None
                continue
            if line.startswith("physical id"):
                phys_id = line.split(":")[-1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":")[-1].strip()
    if phys_id is not None and core_id is not None:
        pairs.add((phys_id, core_id))
    if pairs:
        return len(pairs)
    return None


def _macos_physical_cores() -> int | None:
    # sysctl -n hw.physicalcpu
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
        v = int(out)
        return v if v > 0 else None
    except Exception:
        return None


def _windows_physical_cores() -> int | None:
    # Prefer PowerShell CIM query (wmic is deprecated on newer Windows)
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        v = int(out)
        return v if v > 0 else None
    except Exception:
        return None


def get_memory_info(psutil_mod) -> tuple[int | None, int | None]:
    total = None
    avail = None

    if psutil_mod is not None:
        try:
            vm = psutil_mod.virtual_memory()
            total = int(vm.total)
            avail = int(vm.available)
            return total, avail
        except Exception:
            pass

    try:
        if sys.platform.startswith("linux"):
            total, avail = _linux_meminfo()
        elif sys.platform == "darwin":
            total, avail = _macos_meminfo()
        elif os.name == "nt":
            total, avail = _windows_meminfo()
    except Exception:
        total, avail = None, None

    return total, avail


def _linux_meminfo() -> tuple[int | None, int | None]:
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return None, None
    mt = None
    ma = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mt = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                ma = int(line.split()[1]) * 1024
    return mt, ma


def _macos_meminfo() -> tuple[int | None, int | None]:
    # Total: sysctl hw.memsize
    total = None
    avail = None
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        total = int(out)
    except Exception:
        total = None

    # Available: approximate using vm_stat
    # This is a best‑effort estimate.
    try:
        vm = subprocess.check_output(["vm_stat"], text=True)
        page_size = 4096
        for line in vm.splitlines():
            if "page size of" in line and "bytes" in line:
                # e.g. "Mach Virtual Memory Statistics: (page size of 4096 bytes)"
                try:
                    page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
                except Exception:
                    page_size = 4096
                break

        stats = {}
        for line in vm.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip().strip(".")
            v = v.replace(".", "")
            try:
                stats[k] = int(v)
            except Exception:
                pass

        # "Pages free" + "Pages inactive" is a rough proxy for available.
        free_pages = stats.get("Pages free", 0)
        inactive_pages = stats.get("Pages inactive", 0)
        speculative_pages = stats.get("Pages speculative", 0)
        avail = int((free_pages + inactive_pages + speculative_pages) * page_size)
    except Exception:
        avail = None

    return total, avail


def _windows_meminfo() -> tuple[int | None, int | None]:
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
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    return None, None


def get_disk_usage(path: str) -> dict:
    du = shutil.disk_usage(path)
    return {"path": os.path.abspath(path), "total_bytes": int(du.total), "free_bytes": int(du.free), "used_bytes": int(du.used)}


def get_env_snapshot(keys: list[str]) -> dict:
    out = {}
    for k in keys:
        if k in os.environ:
            out[k] = os.environ.get(k)
    return out


# ----------------------------
# Bench workloads
# ----------------------------

def cpu_work(iters: int) -> int:
    """
    Deterministic integer workload: LCG + xorshift-ish mixing.
    Designed to be CPU-bound and pickle-friendly for multiprocessing.
    """
    x = 0x12345678
    a = 1664525
    c = 1013904223
    mask = 0xFFFFFFFF
    for i in range(iters):
        x = (x * a + c + i) & mask
        x ^= (x >> 16)
        x = (x * 2246822519) & mask
        x ^= (x >> 13)
    return x


def time_cpu(iters: int) -> float:
    t0 = time.perf_counter()
    _ = cpu_work(iters)
    t1 = time.perf_counter()
    return t1 - t0


def calibrate_iters(target_seconds: float, max_iters: int, logger: logging.Logger) -> int:
    iters = 200_000
    iters = clamp_int(iters, 50_000, max_iters)
    last_t = None

    for _ in range(12):
        t = time_cpu(iters)
        last_t = t
        if t <= 0:
            iters = clamp_int(iters * 10, 50_000, max_iters)
            continue

        ratio = target_seconds / t
        if 0.80 <= ratio <= 1.25:
            return iters

        # Adjust (avoid wild swings)
        ratio = max(0.25, min(4.0, ratio))
        iters = int(iters * ratio)
        iters = clamp_int(iters, 50_000, max_iters)

    logger.info(f"Calibration reached max loops; using iters={iters} (last_t={last_t:.4f}s)")
    return iters


def run_single_core_bench(iters: int, trials: int) -> dict:
    times = []
    for _ in range(trials):
        times.append(time_cpu(iters))
    med = float(statistics.median(times))
    return {
        "iters": int(iters),
        "trials": int(trials),
        "median_seconds": med,
        "iters_per_second": (iters / med) if med > 0 else None,
        "trial_seconds": [float(x) for x in times],
    }


def run_multi_core_scaling(max_workers: int, iters_per_task: int, workers_list: list[int], logger: logging.Logger) -> dict:
    """
    Keep a single executor alive; for each 'w' submit 'w' tasks.
    This measures realistic throughput when a persistent pool is used (like in our future engine).
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # safest cross‑platform

    results = []
    if max_workers <= 1:
        return {"max_workers": int(max_workers), "iters_per_task": int(iters_per_task), "results": results}

    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            # Warm-up (forces process spin-up)
            fut = ex.submit(cpu_work, 50_000)
            _ = fut.result(timeout=120)

            for w in workers_list:
                w = int(w)
                if w < 1:
                    continue
                t0 = time.perf_counter()
                futures = [ex.submit(cpu_work, iters_per_task) for _ in range(w)]
                # Wait for completion
                for f in as_completed(futures):
                    _ = f.result()
                t1 = time.perf_counter()

                elapsed = t1 - t0
                total_iters = iters_per_task * w
                itps = (total_iters / elapsed) if elapsed > 0 else None
                results.append(
                    {
                        "workers": w,
                        "elapsed_seconds": float(elapsed),
                        "total_iters": int(total_iters),
                        "iters_per_second_total": itps,
                    }
                )
    except Exception as e:
        logger.exception("Multi-core benchmark failed")
        return {
            "max_workers": int(max_workers),
            "iters_per_task": int(iters_per_task),
            "error": repr(e),
            "results": results,
        }

    return {"max_workers": int(max_workers), "iters_per_task": int(iters_per_task), "results": results}


def run_disk_bench(target_dir: str, file_size_bytes: int, chunk_bytes: int, logger: logging.Logger) -> dict:
    """
    Sequential write + read benchmark.
    """
    ensure_dir(target_dir)

    # Cap by free space (keep big safety margin)
    du = shutil.disk_usage(target_dir)
    free = int(du.free)
    safety = 512 * 1024 * 1024  # keep 512MB free buffer at least
    cap = max(0, free - safety)
    if cap <= 0:
        return {"error": f"Not enough free disk space in {target_dir}: free={human_bytes(free)}"}

    desired = int(file_size_bytes)
    actual_size = min(desired, int(cap * 0.15))  # max 15% of free space
    actual_size = max(min(actual_size, desired), min(desired, 32 * 1024 * 1024))  # at least 32MB if possible
    actual_size = min(actual_size, cap)

    # Prepare reusable chunk (don’t include RNG cost in disk timing)
    chunk = (b"\xA5" * chunk_bytes)

    fd, tmp_path = tempfile.mkstemp(prefix="bench_io_", suffix=".bin", dir=target_dir)
    os.close(fd)

    write_sec = None
    read_sec = None
    write_mb_s = None
    read_mb_s = None

    try:
        # WRITE
        bytes_written = 0
        t0 = time.perf_counter()
        with open(tmp_path, "wb", buffering=0) as f:
            while bytes_written < actual_size:
                n = min(chunk_bytes, actual_size - bytes_written)
                f.write(chunk[:n])
                bytes_written += n
            f.flush()
            os.fsync(f.fileno())
        t1 = time.perf_counter()
        write_sec = t1 - t0
        write_mb_s = (actual_size / (1024 * 1024)) / write_sec if write_sec and write_sec > 0 else None

        # READ
        bytes_read = 0
        t2 = time.perf_counter()
        with open(tmp_path, "rb", buffering=0) as f:
            while True:
                b = f.read(chunk_bytes)
                if not b:
                    break
                bytes_read += len(b)
        t3 = time.perf_counter()
        read_sec = t3 - t2
        read_mb_s = (bytes_read / (1024 * 1024)) / read_sec if read_sec and read_sec > 0 else None

        return {
            "dir": os.path.abspath(target_dir),
            "file_size_bytes": int(actual_size),
            "chunk_bytes": int(chunk_bytes),
            "write_seconds": float(write_sec),
            "write_mb_s": write_mb_s,
            "read_seconds": float(read_sec),
            "read_mb_s": read_mb_s,
        }
    except Exception as e:
        logger.exception("Disk benchmark failed")
        return {"dir": os.path.abspath(target_dir), "error": repr(e)}
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def run_numpy_bench(np, mode: str) -> dict:
    """
    Optional: minimal NumPy throughput hint.
    We avoid gigantic allocations to keep this stable.
    """
    # Size choices: quick vs full
    if mode == "full":
        n = 8_000_000  # ~64MB float64 vector
        reps = 8
    else:
        n = 3_000_000
        reps = 5

    a = np.linspace(0.0, 1.0, n, dtype=np.float64)
    b = np.linspace(1.0, 2.0, n, dtype=np.float64)

    # Warm-up
    c = a * 1.0000001 + b * 0.9999999
    _ = float(c[0])

    t0 = time.perf_counter()
    s = 0.0
    for _ in range(reps):
        c = a * 1.0000001 + b * 0.9999999
        s += float(c[0])
    t1 = time.perf_counter()

    elapsed = t1 - t0
    bytes_touched = reps * n * 8 * 3  # rough: a,b read + c write (float64)
    gb_s = (bytes_touched / (1024**3)) / elapsed if elapsed > 0 else None

    return {
        "numpy_version": getattr(np, "__version__", "unknown"),
        "vector_len": int(n),
        "reps": int(reps),
        "elapsed_seconds": float(elapsed),
        "approx_effective_gb_s": gb_s,
        "sentinel": float(s),
    }


# ----------------------------
# Recommendations (initial heuristic)
# ----------------------------

def recommend_workers(logical_cores: int | None, scaling: dict | None, logger: logging.Logger) -> dict:
    if not logical_cores or logical_cores < 1:
        return {"recommended_cpu_workers": 1, "max_cpu_workers": 1, "note": "os.cpu_count unavailable"}

    max_cpu = int(logical_cores)

    # Default: full usage, but keep option to be conservative later
    recommended = max_cpu

    peak_workers = None
    near_peak_workers = None

    if scaling and "results" in scaling and isinstance(scaling["results"], list) and scaling["results"]:
        rows = [r for r in scaling["results"] if r.get("iters_per_second_total") is not None]
        if rows:
            rows_sorted = sorted(rows, key=lambda r: r["iters_per_second_total"], reverse=True)
            peak = rows_sorted[0]
            peak_thr = float(peak["iters_per_second_total"])
            peak_workers = int(peak["workers"])

            # Choose smallest worker count within 97% of peak throughput (less overhead, similar speed)
            threshold = 0.97 * peak_thr
            candidate = None
            for r in sorted(rows, key=lambda r: r["workers"]):
                thr = r.get("iters_per_second_total")
                if thr is not None and float(thr) >= threshold:
                    candidate = int(r["workers"])
                    break
            near_peak_workers = candidate if candidate is not None else peak_workers

            # For Strategy‑Miner, near-peak is usually the sweet spot.
            recommended = near_peak_workers

    # Clamp
    recommended = clamp_int(int(recommended), 1, max_cpu)

    return {
        "max_cpu_workers": max_cpu,
        "recommended_cpu_workers": recommended,
        "cpu_workers_peak_throughput": peak_workers,
        "cpu_workers_near_peak": near_peak_workers,
    }


def recommend_ram_budget_gb(total_bytes: int | None) -> dict:
    if not total_bytes or total_bytes <= 0:
        return {"ram_total_gb": None, "ram_budget_gb": None, "ram_reserve_gb": None}

    total_gb = total_bytes / (1024**3)

    # Reserve more on small machines, less (relative) on huge machines
    reserve_gb = 3.0 if total_gb < 16 else 4.0
    budget = max(1.0, (total_gb * 0.70) - (reserve_gb * 0.25))
    budget = min(budget, total_gb - reserve_gb)
    budget = max(1.0, budget)

    return {
        "ram_total_gb": round(total_gb, 2),
        "ram_reserve_gb": round(reserve_gb, 2),
        "ram_budget_gb": round(budget, 2),
    }


def recommend_io_workers(disk_bench: dict | None) -> dict:
    # Heuristic: fast disks can benefit from slightly more parallel I/O tasks
    if not disk_bench or disk_bench.get("read_mb_s") is None:
        return {"recommended_io_workers": 2}

    read_mb_s = float(disk_bench["read_mb_s"])
    if read_mb_s >= 1500:
        return {"recommended_io_workers": 6}
    if read_mb_s >= 800:
        return {"recommended_io_workers": 4}
    return {"recommended_io_workers": 2}


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Strategy‑Miner Patch 0001: System benchmark + capability logging")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Quick benchmark (default)")
    mode.add_argument("--full", action="store_true", help="Fuller benchmark (more thorough, longer)")

    parser.add_argument("--disk-path", type=str, default=".", help="Directory used for disk IO benchmark (default: current dir)")
    parser.add_argument("--no-disk", action="store_true", help="Skip disk benchmark")
    parser.add_argument("--no-multi", action="store_true", help="Skip multi-core scaling benchmark")
    args = parser.parse_args()

    bench_mode = "full" if args.full else "quick"

    run_ts = local_ts()
    run_id = short_run_id()
    logger, latest_log, run_log = setup_logger(run_ts, run_id)

    logger.info("============================================================")
    logger.info("Strategy‑Miner — Patch 0001 — System Capability Benchmark")
    logger.info(f"run_ts={run_ts} run_id={run_id} mode={bench_mode}")
    logger.info("============================================================")

    psutil_mod = try_import_psutil()
    np = try_import_numpy()

    # Fingerprint
    env_keys = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
    ]

    logical, physical = get_cpu_counts(psutil_mod)
    mem_total, mem_avail = get_memory_info(psutil_mod)

    disk_usage = None
    try:
        disk_usage = get_disk_usage(args.disk_path)
    except Exception:
        disk_usage = None

    info = {
        "meta": {
            "project": "Strategy‑Miner",
            "patch": "0001",
            "created_utc": utc_now_iso(),
            "mode": bench_mode,
            "run_ts_local": run_ts,
            "run_id": run_id,
        },
        "system": {
            "platform": platform.platform(),
            "os_name": os.name,
            "sys_platform": sys.platform,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": {
            "logical_cores": logical,
            "physical_cores": physical,
        },
        "memory": {
            "total_bytes": mem_total,
            "available_bytes": mem_avail,
        },
        "disk": disk_usage,
        "env": get_env_snapshot(env_keys),
        "optional_libs": {
            "psutil": bool(psutil_mod is not None),
            "numpy": bool(np is not None),
        },
        "benchmarks": {},
        "recommendations": {},
        "artifacts": {
            "log_latest": os.path.abspath(latest_log),
            "log_run": os.path.abspath(run_log),
        },
    }

    logger.info("System snapshot:")
    logger.info(f"  OS: {info['system']['platform']}")
    logger.info(f"  Python: {info['system']['python_version']} ({info['system']['python_implementation']})")
    logger.info(f"  CPU cores: logical={logical} physical={physical}")
    logger.info(f"  RAM: total={human_bytes(mem_total)} available={human_bytes(mem_avail)}")
    if disk_usage:
        logger.info(f"  Disk@{disk_usage['path']}: total={human_bytes(disk_usage['total_bytes'])} free={human_bytes(disk_usage['free_bytes'])}")
    logger.info(f"  Optional: psutil={info['optional_libs']['psutil']} numpy={info['optional_libs']['numpy']}")

    # ----------------------------
    # CPU single-core
    # ----------------------------
    if bench_mode == "full":
        target = 1.00
        trials = 5
        max_iters = 200_000_000
    else:
        target = 0.45
        trials = 3
        max_iters = 120_000_000

    logger.info("")
    logger.info("CPU single-core benchmark:")
    iters = calibrate_iters(target_seconds=target, max_iters=max_iters, logger=logger)
    sc = run_single_core_bench(iters=iters, trials=trials)
    info["benchmarks"]["cpu_single_core"] = sc

    logger.info(f"  calibrated iters={iters}")
    logger.info(f"  median_time={sc['median_seconds']:.4f}s  iters/s={sc['iters_per_second']:.0f}")

    # ----------------------------
    # CPU multi-core scaling
    # ----------------------------
    scaling = None
    if not args.no_multi:
        logger.info("")
        logger.info("CPU multi-core scaling benchmark:")
        max_workers = int(logical or 1)
        if max_workers <= 1:
            logger.info("  skipped (only 1 logical core detected)")
        else:
            if bench_mode == "full":
                # powers of two up to max + max
                ws = []
                w = 1
                while w < max_workers:
                    ws.append(w)
                    w *= 2
                if max_workers not in ws:
                    ws.append(max_workers)
                workers_list = ws
            else:
                # compact set: 1,2,4, max (if available)
                workers_list = [1]
                if max_workers >= 2:
                    workers_list.append(2)
                if max_workers >= 4:
                    workers_list.append(4)
                if max_workers not in workers_list:
                    workers_list.append(max_workers)

            logger.info(f"  workers_list={workers_list}  iters_per_task={iters}")
            scaling = run_multi_core_scaling(
                max_workers=max_workers,
                iters_per_task=iters,
                workers_list=workers_list,
                logger=logger
            )
            info["benchmarks"]["cpu_multi_core_scaling"] = scaling

            # Log summary table
            rows = scaling.get("results", []) if isinstance(scaling, dict) else []
            if rows:
                base = None
                for r in rows:
                    if r.get("workers") == 1 and r.get("iters_per_second_total") is not None:
                        base = float(r["iters_per_second_total"])
                        break
                logger.info("  Results:")
                for r in rows:
                    w = r["workers"]
                    thr = r.get("iters_per_second_total")
                    if thr is None:
                        logger.info(f"    w={w:>2}: throughput=n/a")
                        continue
                    thr = float(thr)
                    speedup = (thr / base) if base and base > 0 else None
                    eff = (speedup / w) if speedup else None
                    logger.info(
                        f"    w={w:>2}: iters/s_total={thr:,.0f}  speedup={speedup:.2f}  eff={eff:.2f}"
                        if speedup is not None and eff is not None
                        else f"    w={w:>2}: iters/s_total={thr:,.0f}"
                    )

    # ----------------------------
    # Disk IO
    # ----------------------------
    disk_bench = None
    if not args.no_disk:
        logger.info("")
        logger.info("Disk sequential IO benchmark:")
        if bench_mode == "full":
            file_size = 512 * 1024 * 1024  # 512 MB
            chunk = 8 * 1024 * 1024        # 8 MB
        else:
            file_size = 128 * 1024 * 1024  # 128 MB
            chunk = 8 * 1024 * 1024        # 8 MB

        disk_bench = run_disk_bench(target_dir=args.disk_path, file_size_bytes=file_size, chunk_bytes=chunk, logger=logger)
        info["benchmarks"]["disk_sequential_io"] = disk_bench

        if disk_bench.get("error"):
            logger.info(f"  disk bench error: {disk_bench['error']}")
        else:
            logger.info(f"  file_size={human_bytes(disk_bench['file_size_bytes'])} chunk={human_bytes(disk_bench['chunk_bytes'])}")
            logger.info(f"  write: {disk_bench['write_mb_s']:.1f} MB/s in {disk_bench['write_seconds']:.3f}s")
            logger.info(f"  read : {disk_bench['read_mb_s']:.1f} MB/s in {disk_bench['read_seconds']:.3f}s")

    # ----------------------------
    # NumPy (optional)
    # ----------------------------
    if np is not None:
        logger.info("")
        logger.info("NumPy vectorization hint benchmark:")
        try:
            nb = run_numpy_bench(np, bench_mode)
            info["benchmarks"]["numpy_vector_hint"] = nb
            gb_s = nb.get("approx_effective_gb_s")
            if gb_s is not None:
                logger.info(f"  numpy={nb['numpy_version']}  approx_effective={gb_s:.2f} GB/s  elapsed={nb['elapsed_seconds']:.3f}s")
            else:
                logger.info(f"  numpy={nb['numpy_version']}  elapsed={nb['elapsed_seconds']:.3f}s")
        except Exception:
            logger.exception("NumPy benchmark failed")
            info["benchmarks"]["numpy_vector_hint"] = {"error": "failed"}

    # ----------------------------
    # Recommendations
    # ----------------------------
    rec = {}
    rec.update(recommend_workers(logical_cores=logical, scaling=scaling, logger=logger))
    rec.update(recommend_ram_budget_gb(mem_total))
    rec.update(recommend_io_workers(disk_bench))
    info["recommendations"] = rec

    logger.info("")
    logger.info("Recommendations (initial heuristic, used by next patches):")
    logger.info(f"  max_cpu_workers={rec.get('max_cpu_workers')}  recommended_cpu_workers={rec.get('recommended_cpu_workers')}")
    if rec.get("ram_budget_gb") is not None:
        logger.info(f"  ram_total_gb={rec.get('ram_total_gb')}  ram_budget_gb={rec.get('ram_budget_gb')}  ram_reserve_gb={rec.get('ram_reserve_gb')}")
    logger.info(f"  recommended_io_workers={rec.get('recommended_io_workers')}")

    # ----------------------------
    # Persist JSON artifacts
    # ----------------------------
    info["artifacts"]["capabilities_latest_json"] = os.path.abspath("capabilities_latest.json")
    info["artifacts"]["capabilities_run_json"] = os.path.abspath(f"capabilities_{run_ts}_{run_id}.json")

    try:
        atomic_write_json(f"capabilities_{run_ts}_{run_id}.json", info)
        atomic_write_json("capabilities_latest.json", info)
        logger.info("")
        logger.info("Artifacts written:")
        logger.info(f"  - {info['artifacts']['capabilities_latest_json']}")
        logger.info(f"  - {info['artifacts']['capabilities_run_json']}")
        logger.info(f"  - {info['artifacts']['log_latest']}")
        logger.info(f"  - {info['artifacts']['log_run']}")
    except Exception:
        logger.exception("Failed writing capability JSON")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
