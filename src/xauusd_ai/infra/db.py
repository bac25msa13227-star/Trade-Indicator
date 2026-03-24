"""PostgreSQL layer — SQLAlchemy 2.x (sync) + psycopg2.

Tables:
  trades          – live closed trades (replaces live_closed_trades.csv)
  signals         – paper/live signal log (replaces paper_trade_signals.csv)
  live_status     – latest live trading status per account (replaces live_status.json)
  learning_events – self-learner events (replaces live_learning_log.jsonl)
  backtest_trades – backtest per-trade results
  walkforward_trades – walkforward per-trade results

Usage:
    from xauusd_ai.infra.db import get_engine, init_tables, TradeStore
    engine = get_engine()
    init_tables(engine)
    store = TradeStore(engine)
    store.upsert_live_status("acc1", {...})
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
    UniqueConstraint,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

_METADATA = MetaData()

# ── Table Definitions ─────────────────────────────────────────────────────────

trades_table = Table(
    "trades",
    _METADATA,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("account", String(32), nullable=False),
    Column("ticket", BigInteger, nullable=False),
    Column("time", DateTime, nullable=False),
    Column("side", String(8)),
    Column("volume", Float),
    Column("open_price", Float),
    Column("close_price", Float),
    Column("profit", Float),
    Column("swap", Float),
    Column("commission", Float),
    Column("pnl", Float),
    Column("is_win", Boolean),
    Column("close_type", String(32)),
    Column("session_id", String(64)),
    Column("inserted_at", DateTime, default=datetime.utcnow),
    UniqueConstraint("account", "ticket", name="uq_trades_account_ticket"),
    Index("ix_trades_account_time", "account", "time"),
)

signals_table = Table(
    "signals",
    _METADATA,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("account", String(32), nullable=False),
    Column("time", DateTime, nullable=False),
    Column("should_trade", Boolean),
    Column("side", String(8)),
    Column("confidence", Float),
    Column("reason", Text),
    Column("entry_price", Float),
    Column("stop_loss", Float),
    Column("take_profit", Float),
    Column("volume", Float),
    Column("strategy_score", Float),
    Column("volatility_regime", Integer),
    Column("account_balance", Float),
    Column("open_positions", Integer),
    Column("inserted_at", DateTime, default=datetime.utcnow),
    Index("ix_signals_account_time", "account", "time"),
)

live_status_table = Table(
    "live_status",
    _METADATA,
    Column("account", String(32), primary_key=True),
    Column("ts", DateTime),
    Column("bar_time", DateTime),
    Column("account_balance", Float),
    Column("open_positions", Integer),
    Column("max_positions", Integer),
    Column("volatility_regime", Integer),
    Column("confidence", Float),
    Column("should_trade", Boolean),
    Column("side", String(8)),
    Column("reason", Text),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

learning_events_table = Table(
    "learning_events",
    _METADATA,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("account", String(32), nullable=False),
    Column("timestamp", DateTime, nullable=False),
    Column("event_type", String(32)),
    Column("loss_count", Integer),
    Column("metrics", Text),   # JSON blob
    Column("inserted_at", DateTime, default=datetime.utcnow),
    Index("ix_learning_account_ts", "account", "timestamp"),
)

backtest_trades_table = Table(
    "backtest_trades",
    _METADATA,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),        # MLflow run_id
    Column("account", String(32)),
    Column("time", DateTime),
    Column("side", String(8)),
    Column("entry_price", Float),
    Column("exit_price", Float),
    Column("pnl", Float),
    Column("is_win", Boolean),
    Column("bars_held", Integer),
    Column("confidence", Float),
    Column("regime", Integer),
    Column("inserted_at", DateTime, default=datetime.utcnow),
    Index("ix_bt_trades_run_id", "run_id"),
)

walkforward_trades_table = Table(
    "walkforward_trades",
    _METADATA,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False),
    Column("fold", Integer),
    Column("account", String(32)),
    Column("time", DateTime),
    Column("side", String(8)),
    Column("entry_price", Float),
    Column("exit_price", Float),
    Column("pnl", Float),
    Column("is_win", Boolean),
    Column("bars_held", Integer),
    Column("confidence", Float),
    Column("inserted_at", DateTime, default=datetime.utcnow),
    Index("ix_wf_trades_run_id_fold", "run_id", "fold"),
)


# ── Engine factory ────────────────────────────────────────────────────────────

def get_db_url() -> str:
    """Build PostgreSQL DSN from environment variables."""
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "trader")
    password = os.getenv("POSTGRES_PASSWORD", "trader_secret")
    db = os.getenv("POSTGRES_DB", "tradedb")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_engine(pool_size: int = 5, max_overflow: int = 10) -> Engine:
    url = get_db_url()
    return create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        echo=False,
    )


def get_engine_no_pool() -> Engine:
    """NullPool engine for Airflow tasks (each task manages its own connection)."""
    return create_engine(get_db_url(), poolclass=NullPool, pool_pre_ping=True)


def init_tables(engine: Engine) -> None:
    """Create all tables if they do not exist."""
    _METADATA.create_all(engine)
    logger.info("PostgreSQL tables initialized.")


@contextmanager
def session_scope(engine: Engine):
    """Provide a transactional scope around a series of operations."""
    conn = engine.connect()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()


# ── Data Store ────────────────────────────────────────────────────────────────

class TradeStore:
    """High-level CRUD operations for trade/signal data."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ── Live status ──────────────────────────────────────────────────────────

    def upsert_live_status(self, account: str, status: dict[str, Any]) -> None:
        row = {
            "account": account,
            "ts": _parse_dt(status.get("ts")),
            "bar_time": _parse_dt(status.get("bar_time")),
            "account_balance": status.get("account_balance"),
            "open_positions": status.get("open_positions"),
            "max_positions": status.get("max_positions"),
            "volatility_regime": status.get("volatility_regime"),
            "confidence": status.get("confidence"),
            "should_trade": status.get("should_trade"),
            "side": status.get("side"),
            "reason": status.get("reason"),
            "updated_at": datetime.utcnow(),
        }
        with session_scope(self.engine) as conn:
            conn.execute(
                text("""
                    INSERT INTO live_status
                      (account, ts, bar_time, account_balance, open_positions,
                       max_positions, volatility_regime, confidence, should_trade,
                       side, reason, updated_at)
                    VALUES
                      (:account, :ts, :bar_time, :account_balance, :open_positions,
                       :max_positions, :volatility_regime, :confidence, :should_trade,
                       :side, :reason, :updated_at)
                    ON CONFLICT (account) DO UPDATE SET
                      ts = EXCLUDED.ts,
                      bar_time = EXCLUDED.bar_time,
                      account_balance = EXCLUDED.account_balance,
                      open_positions = EXCLUDED.open_positions,
                      max_positions = EXCLUDED.max_positions,
                      volatility_regime = EXCLUDED.volatility_regime,
                      confidence = EXCLUDED.confidence,
                      should_trade = EXCLUDED.should_trade,
                      side = EXCLUDED.side,
                      reason = EXCLUDED.reason,
                      updated_at = EXCLUDED.updated_at
                """),
                row,
            )

    def get_live_status(self, account: str) -> dict[str, Any] | None:
        with session_scope(self.engine) as conn:
            result = conn.execute(
                text("SELECT * FROM live_status WHERE account = :account"),
                {"account": account},
            ).mappings().first()
            return dict(result) if result else None

    # ── Trades ───────────────────────────────────────────────────────────────

    def insert_trade(self, account: str, trade: dict[str, Any]) -> None:
        row = {**trade, "account": account, "inserted_at": datetime.utcnow()}
        with session_scope(self.engine) as conn:
            conn.execute(
                text("""
                    INSERT INTO trades
                      (account, ticket, time, side, volume, open_price, close_price,
                       profit, swap, commission, pnl, is_win, close_type, session_id, inserted_at)
                    VALUES
                      (:account, :ticket, :time, :side, :volume, :open_price, :close_price,
                       :profit, :swap, :commission, :pnl, :is_win, :close_type, :session_id, :inserted_at)
                    ON CONFLICT (account, ticket) DO NOTHING
                """),
                row,
            )

    def bulk_insert_trades(self, account: str, df: pd.DataFrame) -> int:
        """Insert trades from DataFrame, returns number of rows inserted."""
        if df.empty:
            return 0
        records = df.copy()
        records["account"] = account
        records["inserted_at"] = datetime.utcnow()
        with session_scope(self.engine) as conn:
            # Use INSERT ... ON CONFLICT DO NOTHING for idempotency
            inserted = 0
            for row in records.to_dict("records"):
                result = conn.execute(
                    text("""
                        INSERT INTO trades
                          (account, ticket, time, side, volume, open_price, close_price,
                           profit, swap, commission, pnl, is_win, close_type, session_id, inserted_at)
                        VALUES
                          (:account, :ticket, :time, :side, :volume, :open_price, :close_price,
                           :profit, :swap, :commission, :pnl, :is_win, :close_type, :session_id, :inserted_at)
                        ON CONFLICT (account, ticket) DO NOTHING
                    """),
                    row,
                )
                inserted += result.rowcount
        return inserted

    def get_trades(
        self,
        account: str,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        query = "SELECT * FROM trades WHERE account = :account"
        params: dict[str, Any] = {"account": account}
        if since:
            query += " AND time >= :since"
            params["since"] = since
        query += " ORDER BY time DESC LIMIT :limit"
        params["limit"] = limit
        with session_scope(self.engine) as conn:
            result = conn.execute(text(query), params)
            rows = result.mappings().all()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    # ── Signals ──────────────────────────────────────────────────────────────

    def insert_signal(self, account: str, signal: dict[str, Any]) -> None:
        row = {**signal, "account": account, "inserted_at": datetime.utcnow()}
        with session_scope(self.engine) as conn:
            conn.execute(
                text("""
                    INSERT INTO signals
                      (account, time, should_trade, side, confidence, reason,
                       entry_price, stop_loss, take_profit, volume, strategy_score,
                       volatility_regime, account_balance, open_positions, inserted_at)
                    VALUES
                      (:account, :time, :should_trade, :side, :confidence, :reason,
                       :entry_price, :stop_loss, :take_profit, :volume, :strategy_score,
                       :volatility_regime, :account_balance, :open_positions, :inserted_at)
                """),
                row,
            )

    def get_signals(self, account: str, limit: int = 500) -> pd.DataFrame:
        with session_scope(self.engine) as conn:
            result = conn.execute(
                text("SELECT * FROM signals WHERE account = :account ORDER BY time DESC LIMIT :limit"),
                {"account": account, "limit": limit},
            )
            rows = result.mappings().all()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    # ── Learning events ──────────────────────────────────────────────────────

    def insert_learning_event(self, account: str, event: dict[str, Any]) -> None:
        row = {
            "account": account,
            "timestamp": _parse_dt(event.get("timestamp", datetime.utcnow().isoformat())),
            "event_type": event.get("event_type", "retrain"),
            "loss_count": event.get("loss_count"),
            "metrics": json.dumps(event.get("metrics", {})),
            "inserted_at": datetime.utcnow(),
        }
        with session_scope(self.engine) as conn:
            conn.execute(
                text("""
                    INSERT INTO learning_events
                      (account, timestamp, event_type, loss_count, metrics, inserted_at)
                    VALUES (:account, :timestamp, :event_type, :loss_count, :metrics, :inserted_at)
                """),
                row,
            )

    # ── Backtest / Walkforward trades ────────────────────────────────────────

    def insert_backtest_trades(self, run_id: str, account: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        records = df.copy()
        records["run_id"] = run_id
        records["account"] = account
        records["inserted_at"] = datetime.utcnow()
        with session_scope(self.engine) as conn:
            for row in records.to_dict("records"):
                conn.execute(
                    text("""
                        INSERT INTO backtest_trades
                          (run_id, account, time, side, entry_price, exit_price,
                           pnl, is_win, bars_held, confidence, regime, inserted_at)
                        VALUES
                          (:run_id, :account, :time, :side, :entry_price, :exit_price,
                           :pnl, :is_win, :bars_held, :confidence, :regime, :inserted_at)
                    """),
                    row,
                )

    def insert_walkforward_trades(
        self, run_id: str, fold: int, account: str, df: pd.DataFrame
    ) -> None:
        if df.empty:
            return
        records = df.copy()
        records["run_id"] = run_id
        records["fold"] = fold
        records["account"] = account
        records["inserted_at"] = datetime.utcnow()
        with session_scope(self.engine) as conn:
            for row in records.to_dict("records"):
                conn.execute(
                    text("""
                        INSERT INTO walkforward_trades
                          (run_id, fold, account, time, side, entry_price, exit_price,
                           pnl, is_win, bars_held, confidence, inserted_at)
                        VALUES
                          (:run_id, :fold, :account, :time, :side, :entry_price, :exit_price,
                           :pnl, :is_win, :bars_held, :confidence, :inserted_at)
                    """),
                    row,
                )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
