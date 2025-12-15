#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner entrypoint (Patch 0003)

Patch 0003 upgrades:
- OOS (train/holdout) evaluation + stronger overfitting filters
- eval_id versioning in SQLite
- show-top defaults to latest eval_id for the dataset
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
from research_engine import ResearchConfig, EvalConfig, eval_id_from_eval_cfg, run_research


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

    # Demo settings
    p.add_argument("--demo-bars", type=int, default=120_000, help="Number of bars for demo dataset")
    p.add_argument("--demo-bar-seconds", type=int, default=60 * 15, help="Bar seconds for demo dataset")
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for search")

    # Eval controls (Patch 0003)
    p.add_argument("--holdout-frac", type=float, default=0.20, help="Holdout fraction (last x%%) for OOS")
    p.add_argument("--min-exposure-train", type=float, default=0.02, help="Min exposure in train slice")
    p.add_argument("--min-exposure-holdout", type=float, default=0.01, help="Min exposure in holdout slice")
    p.add_argument("--min-trades-train", type=int, default=60, help="Min trades in train slice")
    p.add_argument("--min-trades-holdout", type=int, default=10, help="Min trades in holdout slice")
    p.add_argument("--stress-cost-mult", type=float, default=3.0, help="Cost multiplier for stress test")

    p.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild dataset cache even if present")

    # DB browsing
    p.add_argument("--show-top", type=int, default=0, help="Show top N strategies and exit")
    p.add_argument("--dataset-id", type=str, default=None, help="(with --show-top) dataset_id to query")
    p.add_argument("--eval-id", type=str, default=None, help="(optional) eval_id to query. Default: latest for dataset.")
    return p.parse_args()


def _format_top(rows: list[dict], limit: int, eval_id: str | None) -> str:
    if not rows:
        return "(no strategies stored yet for this dataset/eval_id)"
    lines = []
    lines.append(f"eval_id: {eval_id}")
    lines.append(f"Top {min(limit, len(rows))} strategies:")
    lines.append("-" * 140)
    lines.append(
        f"{'rank':>4}  {'score':>8}  {'tr_sh':>7}  {'ho_sh':>7}  {'ho_cal':>7}  {'ho_mdd':>7}  "
        f"{'ho_tr':>6}  {'ho_turn':>7}  {'ho_exp':>6}  hash"
    )
    lines.append("-" * 140)
    for i, r in enumerate(rows[:limit], 1):
        lines.append(
            f"{i:>4}  {r['score']:>8.3f}  "
            f"{float(r.get('train_sharpe') or 0):>7.3f}  {float(r.get('holdout_sharpe') or 0):>7.3f}  "
            f"{float(r.get('holdout_calmar') or 0):>7.3f}  {float(r.get('holdout_max_drawdown') or 0):>7.3f}  "
            f"{int(r.get('trades_holdout') or 0):>6d}  {float(r.get('turnover_holdout') or 0):>7.3f}  {float(r.get('exposure_holdout') or 0):>6.2f}  "
            f"{r['strategy_hash'][:12]}"
        )
    lines.append("-" * 140)
    top = rows[0]
    lines.append("Best genome (compact):")
    lines.append(str(top["genome"]))
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    set_thread_env(threads=1)
    ensure_dir("logs")

    run_tag = local_ts()
    logger = setup_logger(
        "strategy_miner",
        latest_path=os.path.join("logs", "research_latest.log"),
        run_path=os.path.join("logs", f"research_{run_tag}.log"),
    )

    # show-top mode
    if args.show_top and args.show_top > 0:
        if not args.dataset_id:
            print("ERROR: --show-top requires --dataset-id")
            return 2
        eval_id, rows = top_strategies(args.db, str(args.dataset_id), limit=int(args.show_top), eval_id=args.eval_id)
        print(_format_top(rows, int(args.show_top), eval_id))
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

    # Infer bar_seconds for CSV if possible
    bar_seconds = int(cache.meta.get("bar_seconds") or 0)
    if bar_seconds <= 0:
        ts = np.load(cache.ts_path, mmap_mode="r")
        bar_seconds = safe_bar_seconds(ts, default_bar_seconds=int(args.demo_bar_seconds))

    dataset_id = cache.dataset_id

    # Determine worker count
    rec_workers = get_recommended_workers_from_capabilities("capabilities_latest.json")
    auto_workers = rec_workers if rec_workers else (os.cpu_count() or 2)

    workers = int(args.workers) if int(args.workers) > 0 else int(auto_workers)
    workers = max(1, workers)

    # Heuristics for batching/inflight
    batch = int(args.batch) if int(args.batch) > 0 else int(64 if workers <= 4 else 96)
    inflight = int(args.inflight) if int(args.inflight) > 0 else int(max(2, workers * 2))

    # Build eval config
    eval_cfg = EvalConfig(
        holdout_frac=float(args.holdout_frac),
        stress_cost_mult=float(args.stress_cost_mult),
        min_exposure_train=float(args.min_exposure_train),
        min_exposure_holdout=float(args.min_exposure_holdout),
        min_trades_train=int(args.min_trades_train),
        min_trades_holdout=int(args.min_trades_holdout),
    )
    eval_id = eval_id_from_eval_cfg(eval_cfg)

    logger.info("=== Strategy‑Miner (Patch 0003) ===")
    logger.info(f"dataset_id: {dataset_id}")
    logger.info(f"eval_id: {eval_id}")
    logger.info(f"cache.ts_path: {cache.ts_path}")
    logger.info(f"cache.ohlcv_path: {cache.ohlcv_path}")
    logger.info(f"bars: {cache.meta.get('n_bars')} | bar_seconds: {bar_seconds}")
    logger.info(f"db: {args.db}")
    logger.info(f"workers: {workers} | batch: {batch} | inflight: {inflight} | minutes: {args.minutes}")
    logger.info(
        f"holdout_frac={eval_cfg.holdout_frac} | min_exp_train={eval_cfg.min_exposure_train} | "
        f"min_exp_holdout={eval_cfg.min_exposure_holdout} | min_trades_train={eval_cfg.min_trades_train} | "
        f"min_trades_holdout={eval_cfg.min_trades_holdout} | stress_cost_mult={eval_cfg.stress_cost_mult}"
    )

    # DB writer (auto-migrates schema)
    dbw = DBWriter(args.db, commit_every=50, commit_seconds=2.0)
    run_id = f"run_{run_tag}"
    dbw.insert_run(
        run_id=run_id,
        mode="research",
        dataset_id=dataset_id,
        eval_id=eval_id,
        config={
            "minutes": float(args.minutes),
            "workers": workers,
            "batch": batch,
            "inflight": inflight,
            "seed": int(args.seed),
            "eval_id": eval_id,
            "eval_cfg": {
                "holdout_frac": eval_cfg.holdout_frac,
                "min_exposure_train": eval_cfg.min_exposure_train,
                "min_exposure_holdout": eval_cfg.min_exposure_holdout,
                "min_trades_train": eval_cfg.min_trades_train,
                "min_trades_holdout": eval_cfg.min_trades_holdout,
                "stress_cost_mult": eval_cfg.stress_cost_mult,
                "score_version": eval_cfg.score_version,
            },
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
            eval_id=eval_id,
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
        eval_cfg=eval_cfg,
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

    _eval_id, rows = top_strategies(args.db, dataset_id, limit=10, eval_id=eval_id)
    print(_format_top(rows, 10, _eval_id))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
