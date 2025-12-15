#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite persistence for Strategy‑Miner (Patch 0002).

We use WAL mode for concurrent reads and fast single-writer inserts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from sm_utils import ensure_dir, utc_now_iso


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  created_utc TEXT NOT NULL,
  mode TEXT,
  dataset_id TEXT,
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS strategies (
  strategy_hash TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  run_id TEXT NOT NULL,
  genome_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  pass_flags TEXT NOT NULL,
  score REAL NOT NULL,
  sharpe REAL,
  calmar REAL,
  max_drawdown REAL,
  trades INTEGER,
  turnover REAL,
  exposure REAL,
  PRIMARY KEY(strategy_hash, dataset_id)
);

CREATE INDEX IF NOT EXISTS idx_strat_score ON strategies(dataset_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_strat_created ON strategies(created_utc DESC);
"""


def connect(db_path: str) -> sqlite3.Connection:
    ensure_dir(os.path.dirname(os.path.abspath(db_path)) or ".")
    con = sqlite3.connect(db_path, timeout=30.0)
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def init_db(db_path: str) -> None:
    con = connect(db_path)
    try:
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()


class DBWriter:
    """
    Single-writer helper to avoid opening a new sqlite connection per insert.
    """

    def __init__(self, db_path: str, *, commit_every: int = 50, commit_seconds: float = 2.0):
        self.db_path = db_path
        self.con = connect(db_path)
        self.con.executescript(SCHEMA_SQL)
        self.con.commit()
        self.commit_every = int(max(1, commit_every))
        self.commit_seconds = float(max(0.25, commit_seconds))
        self._pending = 0
        self._last_commit = time.time()

    def close(self) -> None:
        try:
            self.con.commit()
        finally:
            self.con.close()

    def maybe_commit(self) -> None:
        now = time.time()
        if self._pending >= self.commit_every or (now - self._last_commit) >= self.commit_seconds:
            self.con.commit()
            self._pending = 0
            self._last_commit = now

    def insert_run(self, run_id: str, mode: str, dataset_id: str, config: dict[str, Any]) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO runs(run_id, created_utc, mode, dataset_id, config_json) VALUES (?,?,?,?,?)",
            (run_id, utc_now_iso(), mode, dataset_id, json.dumps(config, sort_keys=True)),
        )
        self._pending += 1
        self.maybe_commit()

    def upsert_strategy(
        self,
        *,
        strategy_hash: str,
        dataset_id: str,
        run_id: str,
        genome: dict[str, Any],
        metrics: dict[str, Any],
        pass_flags: str,
        score: float,
    ) -> None:
        sharpe = metrics.get("base", {}).get("sharpe")
        calmar = metrics.get("base", {}).get("calmar")
        mdd = metrics.get("base", {}).get("max_drawdown")
        trades = metrics.get("trade", {}).get("trades")
        turnover = metrics.get("trade", {}).get("turnover_mean")
        exposure = metrics.get("trade", {}).get("exposure")

        self.con.execute(
            """
            INSERT INTO strategies(
              strategy_hash, dataset_id, created_utc, run_id,
              genome_json, metrics_json, pass_flags,
              score, sharpe, calmar, max_drawdown, trades, turnover, exposure
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(strategy_hash, dataset_id) DO UPDATE SET
              created_utc=excluded.created_utc,
              run_id=excluded.run_id,
              genome_json=excluded.genome_json,
              metrics_json=excluded.metrics_json,
              pass_flags=excluded.pass_flags,
              score=CASE WHEN excluded.score > strategies.score THEN excluded.score ELSE strategies.score END,
              sharpe=CASE WHEN excluded.score > strategies.score THEN excluded.sharpe ELSE strategies.sharpe END,
              calmar=CASE WHEN excluded.score > strategies.score THEN excluded.calmar ELSE strategies.calmar END,
              max_drawdown=CASE WHEN excluded.score > strategies.score THEN excluded.max_drawdown ELSE strategies.max_drawdown END,
              trades=CASE WHEN excluded.score > strategies.score THEN excluded.trades ELSE strategies.trades END,
              turnover=CASE WHEN excluded.score > strategies.score THEN excluded.turnover ELSE strategies.turnover END,
              exposure=CASE WHEN excluded.score > strategies.score THEN excluded.exposure ELSE strategies.exposure END
            """,
            (
                strategy_hash,
                dataset_id,
                utc_now_iso(),
                run_id,
                json.dumps(genome, sort_keys=True),
                json.dumps(metrics, sort_keys=True),
                pass_flags,
                float(score),
                float(sharpe) if sharpe is not None else None,
                float(calmar) if calmar is not None else None,
                float(mdd) if mdd is not None else None,
                int(trades) if trades is not None else None,
                float(turnover) if turnover is not None else None,
                float(exposure) if exposure is not None else None,
            ),
        )

        self._pending += 1
        self.maybe_commit()


def top_strategies(db_path: str, dataset_id: str, limit: int = 20) -> list[dict[str, Any]]:
    con = connect(db_path)
    try:
        cur = con.execute(
            """
            SELECT strategy_hash, score, sharpe, calmar, max_drawdown, trades, turnover, exposure, created_utc, genome_json
            FROM strategies
            WHERE dataset_id=?
            ORDER BY score DESC
            LIMIT ?
            """,
            (dataset_id, int(limit)),
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "strategy_hash": r[0],
                    "score": r[1],
                    "sharpe": r[2],
                    "calmar": r[3],
                    "max_drawdown": r[4],
                    "trades": r[5],
                    "turnover": r[6],
                    "exposure": r[7],
                    "created_utc": r[8],
                    "genome": json.loads(r[9]),
                }
            )
        return rows
    finally:
        con.close()
