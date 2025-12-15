#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite persistence for Strategy‑Miner (Patch 0003).

Patch 0003 adds:
- eval_id (evaluation config/version hash) so results are comparable & queryable even after we change gating/scoring.
- schema migration from Patch 0002 table layout to the new layout.

We keep WAL mode for speed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from sm_utils import ensure_dir, utc_now_iso


SCHEMA_SQL_V3 = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  created_utc TEXT NOT NULL,
  mode TEXT,
  dataset_id TEXT,
  eval_id TEXT,
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS strategies (
  strategy_hash TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  eval_id TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  run_id TEXT NOT NULL,

  genome_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  pass_flags TEXT NOT NULL,

  score REAL NOT NULL,

  train_sharpe REAL,
  holdout_sharpe REAL,
  train_calmar REAL,
  holdout_calmar REAL,
  train_max_drawdown REAL,
  holdout_max_drawdown REAL,

  trades_train INTEGER,
  trades_holdout INTEGER,
  exposure_train REAL,
  exposure_holdout REAL,
  turnover_train REAL,
  turnover_holdout REAL,

  PRIMARY KEY(strategy_hash, dataset_id, eval_id)
);

CREATE INDEX IF NOT EXISTS idx_strat_score ON strategies(dataset_id, eval_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_strat_created ON strategies(dataset_id, eval_id, created_utc DESC);
"""


def connect(db_path: str) -> sqlite3.Connection:
    ensure_dir(os.path.dirname(os.path.abspath(db_path)) or ".")
    con = sqlite3.connect(db_path, timeout=30.0)
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    cur = con.execute(f"PRAGMA table_info({table});")
    return {str(r[1]) for r in cur.fetchall()}


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def _set_user_version(con: sqlite3.Connection, v: int) -> None:
    con.execute(f"PRAGMA user_version={int(v)};")


def _get_user_version(con: sqlite3.Connection) -> int:
    try:
        cur = con.execute("PRAGMA user_version;")
        return int(cur.fetchone()[0])
    except Exception:
        return 0


def _migrate_from_patch0002(con: sqlite3.Connection) -> None:
    """
    Patch 0002 schema had:
      strategies(strategy_hash, dataset_id) primary key, no eval_id, different columns.
    We rebuild strategies table with eval_id and copy legacy rows with eval_id='legacy_v1'.
    """
    if not _table_exists(con, "strategies"):
        return

    cols = _table_columns(con, "strategies")
    if "eval_id" in cols:
        return  # already migrated

    # Ensure other tables exist
    con.executescript(SCHEMA_SQL_V3)

    # Rename old table
    con.execute("ALTER TABLE strategies RENAME TO strategies_old;")

    # Create new table
    con.executescript("""
    CREATE TABLE IF NOT EXISTS strategies (
      strategy_hash TEXT NOT NULL,
      dataset_id TEXT NOT NULL,
      eval_id TEXT NOT NULL,
      created_utc TEXT NOT NULL,
      run_id TEXT NOT NULL,

      genome_json TEXT NOT NULL,
      metrics_json TEXT NOT NULL,
      pass_flags TEXT NOT NULL,

      score REAL NOT NULL,

      train_sharpe REAL,
      holdout_sharpe REAL,
      train_calmar REAL,
      holdout_calmar REAL,
      train_max_drawdown REAL,
      holdout_max_drawdown REAL,

      trades_train INTEGER,
      trades_holdout INTEGER,
      exposure_train REAL,
      exposure_holdout REAL,
      turnover_train REAL,
      turnover_holdout REAL,

      PRIMARY KEY(strategy_hash, dataset_id, eval_id)
    );

    CREATE INDEX IF NOT EXISTS idx_strat_score ON strategies(dataset_id, eval_id, score DESC);
    CREATE INDEX IF NOT EXISTS idx_strat_created ON strategies(dataset_id, eval_id, created_utc DESC);
    """)

    # Copy legacy data best-effort
    old_cols = _table_columns(con, "strategies_old")

    select_cols = []
    for c in ["strategy_hash","dataset_id","created_utc","run_id","genome_json","metrics_json","pass_flags","score",
              "sharpe","calmar","max_drawdown","trades","turnover","exposure"]:
        if c in old_cols:
            select_cols.append(c)

    if not select_cols:
        con.execute("DROP TABLE strategies_old;")
        return

    cur = con.execute(f"SELECT {', '.join(select_cols)} FROM strategies_old;")
    rows = cur.fetchall()

    for r in rows:
        row = dict(zip(select_cols, r))
        sharpe = row.get("sharpe")
        calmar = row.get("calmar")
        mdd = row.get("max_drawdown")
        trades = row.get("trades")
        turnover = row.get("turnover")
        exposure = row.get("exposure")

        con.execute(
            """
            INSERT OR IGNORE INTO strategies(
              strategy_hash, dataset_id, eval_id, created_utc, run_id,
              genome_json, metrics_json, pass_flags, score,
              train_sharpe, holdout_sharpe, train_calmar, holdout_calmar,
              train_max_drawdown, holdout_max_drawdown,
              trades_train, trades_holdout, exposure_train, exposure_holdout,
              turnover_train, turnover_holdout
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row.get("strategy_hash"),
                row.get("dataset_id"),
                "legacy_v1",
                row.get("created_utc") or utc_now_iso(),
                row.get("run_id") or "legacy",
                row.get("genome_json") or "{}",
                row.get("metrics_json") or "{}",
                row.get("pass_flags") or "legacy",
                float(row.get("score") or 0.0),
                float(sharpe) if sharpe is not None else None,
                float(sharpe) if sharpe is not None else None,
                float(calmar) if calmar is not None else None,
                float(calmar) if calmar is not None else None,
                float(mdd) if mdd is not None else None,
                float(mdd) if mdd is not None else None,
                int(trades) if trades is not None else None,
                int(trades) if trades is not None else None,
                float(exposure) if exposure is not None else None,
                float(exposure) if exposure is not None else None,
                float(turnover) if turnover is not None else None,
                float(turnover) if turnover is not None else None,
            )
        )

    con.execute("DROP TABLE strategies_old;")

    # Runs table may be missing eval_id column in legacy DB
    if _table_exists(con, "runs"):
        run_cols = _table_columns(con, "runs")
        if "eval_id" not in run_cols:
            con.execute("ALTER TABLE runs ADD COLUMN eval_id TEXT;")
            con.execute("UPDATE runs SET eval_id='legacy_v1' WHERE eval_id IS NULL;")

    _set_user_version(con, 3)


def init_db(db_path: str) -> None:
    con = connect(db_path)
    try:
        if _table_exists(con, "strategies"):
            cols = _table_columns(con, "strategies")
            if "eval_id" not in cols:
                _migrate_from_patch0002(con)

        con.executescript(SCHEMA_SQL_V3)
        if _get_user_version(con) < 3:
            _set_user_version(con, 3)
        con.commit()
    finally:
        con.close()


def latest_eval_id(con: sqlite3.Connection, dataset_id: str) -> str | None:
    cur = con.execute(
        """
        SELECT eval_id, MAX(created_utc) AS mx
        FROM strategies
        WHERE dataset_id=?
        GROUP BY eval_id
        ORDER BY mx DESC
        LIMIT 1
        """,
        (dataset_id,),
    )
    row = cur.fetchone()
    return str(row[0]) if row else None


class DBWriter:
    """
    Single-writer helper to avoid opening a new sqlite connection per insert.
    """

    def __init__(self, db_path: str, *, commit_every: int = 50, commit_seconds: float = 2.0):
        self.db_path = db_path
        self.con = connect(db_path)
        init_db(db_path)
        self.con.close()
        self.con = connect(db_path)

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

    def insert_run(self, run_id: str, mode: str, dataset_id: str, eval_id: str, config: dict[str, Any]) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO runs(run_id, created_utc, mode, dataset_id, eval_id, config_json) VALUES (?,?,?,?,?,?)",
            (run_id, utc_now_iso(), mode, dataset_id, str(eval_id), json.dumps(config, sort_keys=True)),
        )
        self._pending += 1
        self.maybe_commit()

    def upsert_strategy(
        self,
        *,
        strategy_hash: str,
        dataset_id: str,
        eval_id: str,
        run_id: str,
        genome: dict[str, Any],
        metrics: dict[str, Any],
        pass_flags: str,
        score: float,
    ) -> None:
        mt = metrics.get("train", {})
        mh = metrics.get("holdout", {})
        tt = metrics.get("train_trade", {})
        th = metrics.get("holdout_trade", {})

        self.con.execute(
            """
            INSERT INTO strategies(
              strategy_hash, dataset_id, eval_id, created_utc, run_id,
              genome_json, metrics_json, pass_flags, score,
              train_sharpe, holdout_sharpe, train_calmar, holdout_calmar,
              train_max_drawdown, holdout_max_drawdown,
              trades_train, trades_holdout, exposure_train, exposure_holdout,
              turnover_train, turnover_holdout
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(strategy_hash, dataset_id, eval_id) DO UPDATE SET
              created_utc=excluded.created_utc,
              run_id=excluded.run_id,
              genome_json=excluded.genome_json,
              metrics_json=excluded.metrics_json,
              pass_flags=excluded.pass_flags,
              score=CASE WHEN excluded.score > strategies.score THEN excluded.score ELSE strategies.score END,

              train_sharpe=CASE WHEN excluded.score > strategies.score THEN excluded.train_sharpe ELSE strategies.train_sharpe END,
              holdout_sharpe=CASE WHEN excluded.score > strategies.score THEN excluded.holdout_sharpe ELSE strategies.holdout_sharpe END,
              train_calmar=CASE WHEN excluded.score > strategies.score THEN excluded.train_calmar ELSE strategies.train_calmar END,
              holdout_calmar=CASE WHEN excluded.score > strategies.score THEN excluded.holdout_calmar ELSE strategies.holdout_calmar END,
              train_max_drawdown=CASE WHEN excluded.score > strategies.score THEN excluded.train_max_drawdown ELSE strategies.train_max_drawdown END,
              holdout_max_drawdown=CASE WHEN excluded.score > strategies.score THEN excluded.holdout_max_drawdown ELSE strategies.holdout_max_drawdown END,

              trades_train=CASE WHEN excluded.score > strategies.score THEN excluded.trades_train ELSE strategies.trades_train END,
              trades_holdout=CASE WHEN excluded.score > strategies.score THEN excluded.trades_holdout ELSE strategies.trades_holdout END,
              exposure_train=CASE WHEN excluded.score > strategies.score THEN excluded.exposure_train ELSE strategies.exposure_train END,
              exposure_holdout=CASE WHEN excluded.score > strategies.score THEN excluded.exposure_holdout ELSE strategies.exposure_holdout END,
              turnover_train=CASE WHEN excluded.score > strategies.score THEN excluded.turnover_train ELSE strategies.turnover_train END,
              turnover_holdout=CASE WHEN excluded.score > strategies.score THEN excluded.turnover_holdout ELSE strategies.turnover_holdout END
            """,
            (
                str(strategy_hash),
                str(dataset_id),
                str(eval_id),
                utc_now_iso(),
                str(run_id),
                json.dumps(genome, sort_keys=True),
                json.dumps(metrics, sort_keys=True),
                str(pass_flags),
                float(score),

                float(mt.get("sharpe")) if mt.get("sharpe") is not None else None,
                float(mh.get("sharpe")) if mh.get("sharpe") is not None else None,
                float(mt.get("calmar")) if mt.get("calmar") is not None else None,
                float(mh.get("calmar")) if mh.get("calmar") is not None else None,
                float(mt.get("max_drawdown")) if mt.get("max_drawdown") is not None else None,
                float(mh.get("max_drawdown")) if mh.get("max_drawdown") is not None else None,

                int(tt.get("trades")) if tt.get("trades") is not None else None,
                int(th.get("trades")) if th.get("trades") is not None else None,
                float(tt.get("exposure")) if tt.get("exposure") is not None else None,
                float(th.get("exposure")) if th.get("exposure") is not None else None,
                float(tt.get("turnover_mean")) if tt.get("turnover_mean") is not None else None,
                float(th.get("turnover_mean")) if th.get("turnover_mean") is not None else None,
            ),
        )

        self._pending += 1
        self.maybe_commit()


def top_strategies(db_path: str, dataset_id: str, *, limit: int = 20, eval_id: str | None = None) -> tuple[str | None, list[dict[str, Any]]]:
    con = connect(db_path)
    try:
        init_db(db_path)
        con.close()
        con = connect(db_path)

        if eval_id is None:
            eval_id = latest_eval_id(con, dataset_id)

        if eval_id is None:
            return None, []

        cur = con.execute(
            """
            SELECT strategy_hash, score,
                   train_sharpe, holdout_sharpe,
                   train_calmar, holdout_calmar,
                   holdout_max_drawdown,
                   trades_holdout, turnover_holdout, exposure_holdout,
                   created_utc, genome_json
            FROM strategies
            WHERE dataset_id=? AND eval_id=?
            ORDER BY score DESC
            LIMIT ?
            """,
            (dataset_id, str(eval_id), int(limit)),
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "strategy_hash": r[0],
                    "score": r[1],
                    "train_sharpe": r[2],
                    "holdout_sharpe": r[3],
                    "train_calmar": r[4],
                    "holdout_calmar": r[5],
                    "holdout_max_drawdown": r[6],
                    "trades_holdout": r[7],
                    "turnover_holdout": r[8],
                    "exposure_holdout": r[9],
                    "created_utc": r[10],
                    "genome": json.loads(r[11]),
                }
            )
        return str(eval_id), rows
    finally:
        con.close()


def load_strategy_genome(db_path: str, dataset_id: str, *, eval_id: str | None, strategy_hash: str) -> tuple[str | None, dict[str, Any] | None]:
    con = connect(db_path)
    try:
        init_db(db_path)
        con.close()
        con = connect(db_path)

        if eval_id is None:
            eval_id = latest_eval_id(con, dataset_id)
        if eval_id is None:
            return None, None

        cur = con.execute(
            """
            SELECT genome_json
            FROM strategies
            WHERE dataset_id=? AND eval_id=? AND strategy_hash=?
            LIMIT 1
            """,
            (dataset_id, str(eval_id), str(strategy_hash)),
        )
        row = cur.fetchone()
        if not row:
            return str(eval_id), None
        return str(eval_id), json.loads(row[0])
    finally:
        con.close()
