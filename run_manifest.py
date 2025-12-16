"""
run_manifest.py

Patch 0007:
- built-in file logging (ohne >> umleiten zu müssen)
- JSON run manifest (args + eval_id + dataset_id + resources + system + git)

Dependency-free (stdlib only) und safe:
- bei Fehlern: NIE crashen
"""
from __future__ import annotations

import datetime as _dt
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Dict, Optional


def _utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _safe_mkdir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _safe_run(cmd: list[str], timeout_s: int = 2) -> str:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
            text=True,
        )
        return (p.stdout or "").strip()
    except Exception:
        return ""


def _cpu_brand() -> str:
    # platform.processor() ist oft leer auf Windows; best-effort per OS.
    try:
        s = platform.processor().strip()
        if s:
            return s
    except Exception:
        pass

    if sys.platform.startswith("win"):
        out = _safe_run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], timeout_s=3)
        out = out.strip()
        if out:
            return out.splitlines()[0].strip()
        out = _safe_run(["wmic", "cpu", "get", "name"], timeout_s=3)
        out = " ".join([ln.strip() for ln in out.splitlines() if ln.strip() and "Name" not in ln])
        return out.strip()

    if sys.platform == "darwin":
        out = _safe_run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout_s=3)
        return out.strip()

    # Linux / others
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if "model name" in ln:
                        return ln.split(":", 1)[1].strip()
    except Exception:
        pass

    return ""


def git_commit_short() -> str:
    return _safe_run(["git", "rev-parse", "--short", "HEAD"], timeout_s=2).strip()


def git_status_porcelain() -> str:
    return _safe_run(["git", "status", "--porcelain"], timeout_s=2).strip()


def setup_run_logging(log_file: Optional[str]) -> None:
    """
    Adds a file sink if loguru is present; otherwise uses stdlib logging.
    Never raises.
    """
    if not log_file:
        return
    try:
        _safe_mkdir_for_file(log_file)
    except Exception:
        pass

    # Prefer loguru if available.
    try:
        from loguru import logger  # type: ignore
        logger.add(
            log_file,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            rotation="50 MB",
            retention=10,
            compression=None,
        )
        return
    except Exception:
        pass

    # Fallback: stdlib logging
    try:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            filename=log_file,
            filemode="a",
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
    except Exception:
        pass


def _jsonify(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return _jsonify(vars(obj))
    return str(obj)


def build_run_manifest(
    *,
    args: Any,
    dataset_id: str,
    eval_id: str,
    resources: Dict[str, Any],
    capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    caps = capabilities or {}
    return {
        "schema": "strategy_miner.run_manifest.v1",
        "created_utc": _utc_iso(),
        "host": {
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "cpu_brand": _cpu_brand(),
            "logical_cores": os.cpu_count() or 0,
        },
        "git": {
            "commit": git_commit_short(),
            "dirty_porcelain": git_status_porcelain(),
        },
        "run": {
            "dataset_id": dataset_id,
            "eval_id": eval_id,
            "resources": _jsonify(resources),
        },
        "args": _jsonify(args),
        "capabilities_latest": _jsonify(caps),
    }


def write_run_manifest(path: str, manifest: Dict[str, Any], *, also_timestamped: bool = True) -> None:
    """
    Write manifest to `path` and optionally to a timestamped sibling file.

    Example:
      path="run_latest.json"
      => run_latest.json + run_20251216T021008Z_eval_xxxx.json
    """
    try:
        _safe_mkdir_for_file(path)
    except Exception:
        pass

    def _write(p: str) -> None:
        _safe_mkdir_for_file(p)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, p)

    try:
        _write(path)
    except Exception:
        return

    if also_timestamped:
        try:
            ts = manifest.get("created_utc", _utc_iso())
            ts_clean = (
                str(ts)
                .replace("-", "")
                .replace(":", "")
                .replace("+00:00", "Z")
            )
            ts_clean = "".join(ch for ch in ts_clean if ch.isalnum() or ch in ("T", "Z"))
            eval_id = str(manifest.get("run", {}).get("eval_id", "eval")).replace(os.sep, "_")
            _write(f"run_{ts_clean}_{eval_id}.json")
        except Exception:
            pass
