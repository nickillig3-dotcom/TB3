#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner entrypoint (Patch 0004)

Patch 0004:
- Adds Deep Validation Stage (time-block CV) controls via CLI
- Prints CV summary in --show-top
- eval_id changes automatically with EvalConfig

Run examples:
  python strategy_miner.py --demo --minutes 0.5 --demo-bars 40000 --cv-folds 5
  python strategy_miner.py --show-top 20 --dataset-id demo_40000_900_1234
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

    # Eval controls (Stage-1)
    p.add_argument("--holdout-frac", type=float, default=0.20, help="Holdout fraction (last x%%) for OOS")
    p.add_argument("--min-exposure-train", type=float, default=0.02, help="Min exposure in train slice")
    p.add_argument("--min-exposure-holdout", type=float, default=0.01, help="Min exposure in holdout slice")
    p.add_argument("--min-trades-train", type=int, default=60, help="Min trades in train slice")
    p.add_argument("--min-trades-holdout", type=int, default=10, help="Min trades in holdout slice")
    p.add_argument("--stress-cost-mult", type=float, default=3.0, help="Cost multiplier for stress test")

    # Deep validation (Stage-2 CV)
    p.add_argument("--cv-folds", type=int, default=5, help="Time-block CV folds (>=2 enables DVS)")
    p.add_argument("--no-deep", action="store_true", help="Disable deep validation stage (CV)")
    p.add_argument("--cv-delay", type=int, default=1, help="Delay bars used in CV delay stress (default 1)")
    p.add_argument("--cv-min-pos-frac", type=float, default=0.60, help="Min fraction of positive folds")
    p.add_argument("--cv-min-sharpe", type=float, default=-0.20, help="Min fold Sharpe allowed (base)")
    p.add_argument("--cv-min-med-base", type=float, default=0.20, help="Min median Sharpe across folds (base)")
    p.add_argument("--cv-min-med-delay", type=float, default=0.10, help="Min median Sharpe across folds (delay)")
    p.add_argument("--cv-min-med-cost", type=float, default=0.05, help="Min median Sharpe across folds (cost stress)")

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
    lines.append("-" * 190)
    lines.append(
        f"{'rank':>4} {'score':>8} "
        f"{'tr_sh':>7} {'ho_sh':>7} {'ho_mdd':>7} "
        f"{'cv_med':>7} {'cv_min':>7} {'cv_d1':>7} {'cv_cost':>7} {'cv_pf':>5} "
        f"{'ho_tr':>6} {'ho_turn':>7} {'ho_exp':>6} hash"
    )
    lines.append("-" * 190)

    for i, r in enumerate(rows[:limit], 1):
        cv_pf = ""
        if r.get("cv_pos_folds") is not None and r.get("cv_n_folds") is not None:
            cv_pf = f"{int(r.get('cv_pos_folds') or 0)}/{int(r.get('cv_n_folds') or 0)}"
        else:
            cv_pf = "n/a"

        lines.append(
            f"{i:>4} {float(r['score']):>8.3f} "
            f"{float(r.get('train_sharpe') or 0):>7.3f} {float(r.get('holdout_sharpe') or 0):>7.3f} {float(r.get('holdout_max_drawdown') or 0):>7.3f} "
            f"{float(r.get('cv_median_sharpe') or 0):>7.3f} {float(r.get('cv_min_sharpe') or 0):>7.3f} "
            f"{float(r.get('cv_med_delay1') or 0):>7.3f} {float(r.get('cv_med_cost') or 0):>7.3f} {cv_pf:>5} "
            f"{int(r.get('trades_holdout') or 0):>6d} {float(r.get('turnover_holdout') or 0):>7.3f} {float(r.get('exposure_holdout') or 0):>6.2f} "
            f"{r['strategy_hash'][:12]}"
        )

    lines.append("-" * 190)
    lines.append("Best genome (compact):")
    lines.append(str(rows[0]["genome"]))
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
        _eval_id, rows = top_strategies(args.db, str(args.dataset_id), limit=int(args.show_top), eval_id=args.eval_id)
        print(_format_top(rows, int(args.show_top), _eval_id))
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

    bar_seconds = int(cache.meta.get("bar_seconds") or 0)
    if bar_seconds <= 0:
        ts = np.load(cache.ts_path, mmap_mode="r")
        bar_seconds = safe_bar_seconds(ts, default_bar_seconds=int(args.demo_bar_seconds))

    dataset_id = cache.dataset_id

    # Worker count
    rec_workers = get_recommended_workers_from_capabilities("capabilities_latest.json")
    auto_workers = rec_workers if rec_workers else (os.cpu_count() or 2)
    workers = int(args.workers) if int(args.workers) > 0 else int(auto_workers)
    workers = max(1, workers)

    batch = int(args.batch) if int(args.batch) > 0 else int(64 if workers <= 4 else 96)
    inflight = int(args.inflight) if int(args.inflight) > 0 else int(max(2, workers * 2))

    # Eval config
    eval_cfg = EvalConfig(
        holdout_frac=float(args.holdout_frac),
        stress_cost_mult=float(args.stress_cost_mult),
        min_exposure_train=float(args.min_exposure_train),
        min_exposure_holdout=float(args.min_exposure_holdout),
        min_trades_train=int(args.min_trades_train),
        min_trades_holdout=int(args.min_trades_holdout),

        deep_validation=(not bool(args.no_deep)),
        cv_folds=int(args.cv_folds),
        cv_delay_bars=int(args.cv_delay),
        cv_min_positive_folds_frac=float(args.cv_min_pos_frac),
        cv_min_sharpe=float(args.cv_min_sharpe),
        cv_min_median_sharpe_base=float(args.cv_min_med_base),
        cv_min_median_sharpe_delay1=float(args.cv_min_med_delay),
        cv_min_median_sharpe_coststress=float(args.cv_min_med_cost),
    )
    eval_id = eval_id_from_eval_cfg(eval_cfg)

    logger.info("=== Strategy‑Miner (Patch 0004) ===")
    logger.info(f"dataset_id: {dataset_id}")
    logger.info(f"eval_id: {eval_id}")
    logger.info(f"bars: {cache.meta.get('n_bars')} | bar_seconds: {bar_seconds}")
    logger.info(f"db: {args.db}")
    logger.info(f"workers: {workers} | batch: {batch} | inflight: {inflight} | minutes: {args.minutes}")
    logger.info(
        f"holdout_frac={eval_cfg.holdout_frac} | stress_cost_mult={eval_cfg.stress_cost_mult} | "
        f"deep_validation={eval_cfg.deep_validation} | cv_folds={eval_cfg.cv_folds} | cv_delay={eval_cfg.cv_delay_bars}"
    )
    logger.info(
        f"cv_min_pos_frac={eval_cfg.cv_min_positive_folds_frac} | cv_min_sharpe={eval_cfg.cv_min_sharpe} | "
        f"cv_min_med_base={eval_cfg.cv_min_median_sharpe_base} | cv_min_med_delay={eval_cfg.cv_min_median_sharpe_delay1} | "
        f"cv_min_med_cost={eval_cfg.cv_min_median_sharpe_coststress}"
    )

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
                "stress_cost_mult": eval_cfg.stress_cost_mult,
                "deep_validation": eval_cfg.deep_validation,
                "cv_folds": eval_cfg.cv_folds,
                "cv_delay_bars": eval_cfg.cv_delay_bars,
                "cv_min_positive_folds_frac": eval_cfg.cv_min_positive_folds_frac,
                "cv_min_sharpe": eval_cfg.cv_min_sharpe,
                "cv_min_median_sharpe_base": eval_cfg.cv_min_median_sharpe_base,
                "cv_min_median_sharpe_delay1": eval_cfg.cv_min_median_sharpe_delay1,
                "cv_min_median_sharpe_coststress": eval_cfg.cv_min_median_sharpe_coststress,
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
