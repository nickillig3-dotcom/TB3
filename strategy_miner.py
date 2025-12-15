#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy‑Miner entrypoint (Patch 0005)

Patch 0005:
- Evolutionary search mode (default) with ElitePool + mutation/crossover
- Elite warmstart loading from DB
- CV v2 gates exposed via CLI
- Regime attribution enabled by default (stored in metrics_json["regime"])
- --show-top prints CV minima + regime coverage/dominance
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

    # Search (Patch 0005)
    p.add_argument("--search-mode", type=str, default="evo", choices=["evo", "random"])
    p.add_argument("--elite-size", type=int, default=512)
    p.add_argument("--elite-load", type=int, default=128, help="Load top N genomes from DB to seed evo search (0 disables)")
    p.add_argument("--seen-max", type=int, default=200_000)

    # Eval controls (Stage-1)
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--min-exposure-train", type=float, default=0.02)
    p.add_argument("--min-exposure-holdout", type=float, default=0.01)
    p.add_argument("--min-trades-train", type=int, default=60)
    p.add_argument("--min-trades-holdout", type=int, default=10)
    p.add_argument("--stress-cost-mult", type=float, default=3.0)

    # Deep validation (Stage-2 CV)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--no-deep", action="store_true")
    p.add_argument("--cv-delay", type=int, default=1)
    p.add_argument("--cv-min-pos-frac", type=float, default=0.60)

    # CV v2 gates (Patch 0005)
    p.add_argument("--cv-min-sharpe", type=float, default=-0.20)
    p.add_argument("--cv-min-med-base", type=float, default=0.20)
    p.add_argument("--cv-min-med-delay", type=float, default=0.10)
    p.add_argument("--cv-min-med-cost", type=float, default=0.05)
    p.add_argument("--cv-min-sharpe-delay1", type=float, default=-0.10)
    p.add_argument("--cv-min-sharpe-cost", type=float, default=-0.10)
    p.add_argument("--cv-min-pos-cost-frac", type=float, default=0.60)

    # Regime attribution
    p.add_argument("--no-regime", action="store_true")
    p.add_argument("--regime-vol-window", type=int, default=160)
    p.add_argument("--regime-trend-window", type=int, default=160)
    p.add_argument("--regime-min-bars", type=int, default=600)
    p.add_argument("--regime-min-exp", type=float, default=0.03)

    p.add_argument("--force-rebuild-cache", action="store_true")

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
    lines.append("-" * 220)
    lines.append(
        f"{'rank':>4} {'score':>8} "
        f"{'tr_sh':>7} {'ho_sh':>7} {'ho_mdd':>7} "
        f"{'cv_med':>7} {'cv_min':>7} {'cv_md1':>7} {'cv_mC':>7} "
        f"{'cv_minD1':>9} {'cv_minC':>8} {'cv_pf':>5} {'c_pf':>5} "
        f"{'reg_cov':>7} {'reg_dom':>7} "
        f"{'ho_tr':>6} {'ho_turn':>7} {'ho_exp':>6} hash"
    )
    lines.append("-" * 220)

    for i, r in enumerate(rows[:limit], 1):
        def _pf(a, b) -> str:
            if a is None or b is None:
                return "n/a"
            return f"{int(a)}/{int(b)}"

        cv_pf = _pf(r.get("cv_pos_folds"), r.get("cv_n_folds"))
        c_pf = _pf(r.get("cv_pos_cost"), r.get("cv_n_folds"))

        reg_cov = r.get("reg_coverage")
        reg_dom = r.get("reg_dominance")

        lines.append(
            f"{i:>4} {float(r['score']):>8.3f} "
            f"{float(r.get('train_sharpe') or 0):>7.3f} {float(r.get('holdout_sharpe') or 0):>7.3f} {float(r.get('holdout_max_drawdown') or 0):>7.3f} "
            f"{float(r.get('cv_median_sharpe') or 0):>7.3f} {float(r.get('cv_min_sharpe') or 0):>7.3f} "
            f"{float(r.get('cv_med_delay1') or 0):>7.3f} {float(r.get('cv_med_cost') or 0):>7.3f} "
            f"{float(r.get('cv_min_delay1') or 0):>9.3f} {float(r.get('cv_min_cost') or 0):>8.3f} "
            f"{cv_pf:>5} {c_pf:>5} "
            f"{int(reg_cov) if reg_cov is not None else 0:>7d} {float(reg_dom) if reg_dom is not None else 0.0:>7.3f} "
            f"{int(r.get('trades_holdout') or 0):>6d} {float(r.get('turnover_holdout') or 0):>7.3f} {float(r.get('exposure_holdout') or 0):>6.2f} "
            f"{r['strategy_hash'][:12]}"
        )

    lines.append("-" * 220)
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

        cv_min_sharpe_delay1=float(args.cv_min_sharpe_delay1),
        cv_min_sharpe_coststress=float(args.cv_min_sharpe_cost),
        cv_min_positive_folds_coststress_frac=float(args.cv_min_pos_cost_frac),

        regime_enabled=(not bool(args.no_regime)),
        regime_vol_window=int(args.regime_vol_window),
        regime_trend_window=int(args.regime_trend_window),
        regime_min_bars=int(args.regime_min_bars),
        regime_min_exposure=float(args.regime_min_exp),
    )
    eval_id = eval_id_from_eval_cfg(eval_cfg)

    # Load elite seeds from DB (optional)
    elite_seeds = []
    elite_seed_eval = None
    if args.search_mode == "evo" and int(args.elite_load) > 0:
        elite_seed_eval, seed_rows = top_strategies(args.db, dataset_id, limit=int(args.elite_load), eval_id=None)
        elite_seeds = [r["genome"] for r in seed_rows] if seed_rows else []

    logger.info("=== Strategy‑Miner (Patch 0005) ===")
    logger.info(f"dataset_id: {dataset_id}")
    logger.info(f"eval_id: {eval_id}")
    if elite_seed_eval:
        logger.info(f"elite_seed_from_eval: {elite_seed_eval} | elite_seed_genomes: {len(elite_seeds)}")

    logger.info(f"bars: {cache.meta.get('n_bars')} | bar_seconds: {bar_seconds}")
    logger.info(f"db: {args.db}")
    logger.info(f"workers: {workers} | batch: {batch} | inflight: {inflight} | minutes: {args.minutes}")
    logger.info(f"search_mode={args.search_mode} | elite_size={args.elite_size} | elite_load={args.elite_load} | seen_max={args.seen_max}")
    logger.info(
        f"holdout_frac={eval_cfg.holdout_frac} | stress_cost_mult={eval_cfg.stress_cost_mult} | "
        f"deep_validation={eval_cfg.deep_validation} | cv_folds={eval_cfg.cv_folds} | cv_delay={eval_cfg.cv_delay_bars}"
    )
    logger.info(
        f"cv_min_pos_frac={eval_cfg.cv_min_positive_folds_frac} | cv_min_pos_cost_frac={eval_cfg.cv_min_positive_folds_coststress_frac} | "
        f"cv_min_sharpe={eval_cfg.cv_min_sharpe} | cv_min_sharpe_delay1={eval_cfg.cv_min_sharpe_delay1} | cv_min_sharpe_cost={eval_cfg.cv_min_sharpe_coststress}"
    )
    logger.info(
        f"regime_enabled={eval_cfg.regime_enabled} | vol_w={eval_cfg.regime_vol_window} | trend_w={eval_cfg.regime_trend_window} | "
        f"reg_min_bars={eval_cfg.regime_min_bars} | reg_min_exp={eval_cfg.regime_min_exposure}"
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
            "search_mode": str(args.search_mode),
            "elite_size": int(args.elite_size),
            "elite_load": int(args.elite_load),
            "seen_max": int(args.seen_max),
            "eval_id": eval_id,
            "eval_cfg": {
                "holdout_frac": eval_cfg.holdout_frac,
                "stress_cost_mult": eval_cfg.stress_cost_mult,
                "deep_validation": eval_cfg.deep_validation,
                "cv_folds": eval_cfg.cv_folds,
                "cv_delay_bars": eval_cfg.cv_delay_bars,
                "cv_min_positive_folds_frac": eval_cfg.cv_min_positive_folds_frac,
                "cv_min_positive_folds_coststress_frac": eval_cfg.cv_min_positive_folds_coststress_frac,
                "cv_min_sharpe": eval_cfg.cv_min_sharpe,
                "cv_min_median_sharpe_base": eval_cfg.cv_min_median_sharpe_base,
                "cv_min_median_sharpe_delay1": eval_cfg.cv_min_median_sharpe_delay1,
                "cv_min_median_sharpe_coststress": eval_cfg.cv_min_median_sharpe_coststress,
                "cv_min_sharpe_delay1": eval_cfg.cv_min_sharpe_delay1,
                "cv_min_sharpe_coststress": eval_cfg.cv_min_sharpe_coststress,
                "regime_enabled": eval_cfg.regime_enabled,
                "regime_vol_window": eval_cfg.regime_vol_window,
                "regime_trend_window": eval_cfg.regime_trend_window,
                "regime_min_bars": eval_cfg.regime_min_bars,
                "regime_min_exposure": eval_cfg.regime_min_exposure,
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
            f"rate={p['tested_per_sec']:.1f}/s | best_score={p['best_score']:.3f} | "
            f"in_flight={p['in_flight']} | elite={p.get('elite_size',0)} | seen={p.get('seen',0)}"
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

        search_mode=str(args.search_mode),
        elite_size=int(args.elite_size),
        elite_seed_genomes=list(elite_seeds),
        seen_max=int(args.seen_max),
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
