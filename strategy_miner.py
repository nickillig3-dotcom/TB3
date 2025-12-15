#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner entrypoint (Patch 0002)

Usage examples:

  # Run on synthetic demo data for 1 minute
  python strategy_miner.py --demo --minutes 1

  # Run for 30 seconds, fewer bars (faster for sanity check)
  python strategy_miner.py --demo --minutes 0.5 --demo-bars 40000

  # Show top strategies from last run (for the dataset_id)
  python strategy_miner.py --show-top 20

  # Use your own CSV (must have at least 'close' column)
  python strategy_miner.py --csv path/to/data.csv --minutes 3

Notes:
- This is research-only; no execution/broker integration.
- The database is local SQLite (strategy_results.sqlite by default).
"""

from __future__ import annotations

import argparse
import os
import time
import multiprocessing as mp

import numpy as np

from sm_utils import (
    ensure_dir,
    local_ts,
    setup_logger,
    set_thread_env,
    get_recommended_workers_from_capabilities,
)
from data_cache import prepare_demo_dataset, prepare_csv_dataset
from bt_eval import safe_bar_seconds
from results_db import DBWriter, top_strategies
from research_engine import ResearchConfig, run_research


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--demo", action="store_true", help="Use synthetic demo dataset")
    src.add_argument("--csv", type=str, default=None, help="Path to CSV with OHLCV columns (needs at least close)")

    p.add_argument("--minutes", type=float, default=1.0, help="How long to run research")
    p.add_argument("--db", type=str, default="strategy_results.sqlite", help="SQLite db path")
    p.add_argument("--workers", type=int, default=0, help="Override worker count (0=auto)")
    p.add_argument("--batch", type=int, default=0, help="Strategies per worker task (0=auto)")
    p.add_argument("--inflight", type=int, default=0, help="Max tasks in flight (0=auto)")

    p.add_argument("--demo-bars", type=int, default=120_000, help="Number of bars for demo dataset")
    p.add_argument("--demo-bar-seconds", type=int, default=60 * 15, help="Bar seconds for demo dataset")
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for search")

    p.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild dataset cache even if present")
    p.add_argument("--show-top", type=int, default=0, help="Show top N strategies and exit")
    p.add_argument("--dataset-id", type=str, default=None, help="(with --show-top) override dataset_id")
    return p.parse_args()


def _format_top(rows: list[dict], limit: int) -> str:
    if not rows:
        return "(no strategies stored yet for this dataset_id)"
    lines = []
    lines.append(f"Top {min(limit, len(rows))} strategies:")
    lines.append("-" * 110)
    lines.append(f"{'rank':>4}  {'score':>8}  {'sharpe':>8}  {'calmar':>8}  {'mdd':>8}  {'trades':>8}  {'turn':>8}  {'exp':>6}  hash")
    lines.append("-" * 110)
    for i, r in enumerate(rows[:limit], 1):
        lines.append(
            f"{i:>4}  {r['score']:>8.3f}  {r.get('sharpe',0):>8.3f}  {r.get('calmar',0):>8.3f}  {r.get('max_drawdown',0):>8.3f}  "
            f"{int(r.get('trades') or 0):>8d}  {float(r.get('turnover') or 0):>8.3f}  {float(r.get('exposure') or 0):>6.2f}  {r['strategy_hash'][:12]}"
        )
    lines.append("-" * 110)
    top = rows[0]
    lines.append("Best genome (compact):")
    lines.append(str(top["genome"]))
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    # Avoid numpy/BLAS oversubscription with process pool
    set_thread_env(threads=1)

    ensure_dir("logs")

    run_tag = local_ts()
    logger = setup_logger(
        "strategy_miner",
        latest_path=os.path.join("logs", "research_latest.log"),
        run_path=os.path.join("logs", f"research_{run_tag}.log"),
    )

    # Show-top mode with explicit dataset-id (no dataset preparation needed)
    if args.show_top and args.show_top > 0 and args.dataset_id:
        rows = top_strategies(args.db, str(args.dataset_id), limit=int(args.show_top))
        print(_format_top(rows, int(args.show_top)))
        return 0

    # Prepare dataset cache
    if args.csv:
        cache = prepare_csv_dataset(args.csv, force_rebuild=args.force_rebuild_cache)
    else:
        cache = prepare_demo_dataset(
            n_bars=int(args.demo_bars),
            seed=int(args.seed),
            bar_seconds=int(args.demo_bar_seconds),
            force_rebuild=args.force_rebuild_cache,
        )

    # infer bar_seconds for CSV if possible
    bar_seconds = int(cache.meta.get("bar_seconds") or 0)
    if bar_seconds <= 0:
        ts = np.load(cache.ts_path, mmap_mode="r")
        bar_seconds = safe_bar_seconds(ts, default_bar_seconds=int(args.demo_bar_seconds))

    dataset_id = cache.dataset_id

    # Show-top mode (derives dataset-id from current dataset selection)
    if args.show_top and args.show_top > 0:
        rows = top_strategies(args.db, dataset_id, limit=int(args.show_top))
        print(_format_top(rows, int(args.show_top)))
        return 0

    # Determine worker count
    rec_workers = get_recommended_workers_from_capabilities("capabilities_latest.json")
    auto_workers = rec_workers if rec_workers else (os.cpu_count() or 2)

    workers = int(args.workers) if int(args.workers) > 0 else int(auto_workers)
    workers = max(1, workers)

    # Heuristics for batching/inflight
    batch = int(args.batch) if int(args.batch) > 0 else int(64 if workers <= 4 else 96)
    inflight = int(args.inflight) if int(args.inflight) > 0 else int(max(2, workers * 2))

    logger.info("=== Strategy‑Miner (Patch 0002) ===")
    logger.info(f"dataset_id: {dataset_id}")
    logger.info(f"cache.ts_path: {cache.ts_path}")
    logger.info(f"cache.ohlcv_path: {cache.ohlcv_path}")
    logger.info(f"bars: {cache.meta.get('n_bars')} | bar_seconds: {bar_seconds}")
    logger.info(f"db: {args.db}")
    logger.info(f"workers: {workers} | batch: {batch} | inflight: {inflight} | minutes: {args.minutes}")

    # DB writer
    dbw = DBWriter(args.db, commit_every=50, commit_seconds=2.0)
    run_id = f"run_{run_tag}"
    dbw.insert_run(
        run_id=run_id,
        mode="research",
        dataset_id=dataset_id,
        config={
            "minutes": float(args.minutes),
            "workers": workers,
            "batch": batch,
            "inflight": inflight,
            "seed": int(args.seed),
        },
    )

    accepted_since = 0
    last_db_report = time.time()

    def on_result(r: dict) -> None:
        nonlocal accepted_since
        accepted_since += 1
        dbw.upsert_strategy(
            strategy_hash=r["strategy_hash"],
            dataset_id=dataset_id,
            run_id=run_id,
            genome=r["genome"],
            metrics=r["metrics"],
            pass_flags=r["pass_flags"],
            score=float(r["score"]),
        )

    def on_progress(p: dict) -> None:
        nonlocal accepted_since, last_db_report
        logger.info(
            f"tested={p['tested']} | acc={p['accepted']} ({p['accepted_pct']:.2f}%) | "
            f"rate={p['tested_per_sec']:.1f}/s | best_score={p['best_score']:.3f} | in_flight={p['in_flight']}"
        )
        now = time.time()
        if now - last_db_report >= 10.0:
            logger.info(f"accepted_last_10s={accepted_since}")
            accepted_since = 0
            last_db_report = now

    cfg = ResearchConfig(
        dataset_id=dataset_id,
        ts_path=cache.ts_path,
        ohlcv_path=cache.ohlcv_path,
        bar_seconds=int(bar_seconds),
        db_path=str(args.db),
        run_minutes=float(args.minutes),
        workers=int(workers),
        batch_size=int(batch),
        max_in_flight=int(inflight),
        seed=int(args.seed),
        mode="research",
    )

    try:
        summary = run_research(cfg, on_result=on_result, on_progress=on_progress)
    finally:
        dbw.close()

    logger.info("=== DONE ===")
    logger.info(
        f"tested={summary['tested']} | accepted={summary['accepted']} ({summary['accepted_pct']:.2f}%) | "
        f"rate={summary['tested_per_sec']:.1f}/s | best_score={summary['best_score']:.3f} | seconds={summary['seconds']:.1f}"
    )

    # Print top 10 after run
    rows = top_strategies(args.db, dataset_id, limit=10)
    print(_format_top(rows, 10))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
