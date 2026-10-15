-- PostgreSQL Schema for Raw Trading Exports Cleaning and Validation Pipeline

-- 1. Raw Staging Tables (unconstrained for initial bulk ingest)
CREATE TABLE IF NOT EXISTS raw_trades_export (
    raw_id BIGSERIAL PRIMARY KEY,
    trade_id TEXT,
    symbol TEXT,
    side TEXT,
    quantity NUMERIC,
    entry_price NUMERIC,
    exit_price NUMERIC,
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    pnl NUMERIC,
    return_pct NUMERIC,
    commission NUMERIC,
    holding_period_bars INTEGER,
    ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_positions_export (
    raw_id BIGSERIAL PRIMARY KEY,
    portfolio_id TEXT,
    symbol TEXT,
    name TEXT,
    asset_class TEXT,
    sector TEXT,
    side TEXT,
    quantity NUMERIC,
    entry_price NUMERIC,
    current_price NUMERIC,
    market_value NUMERIC,
    unrealized_pnl NUMERIC,
    unrealized_pnl_pct NUMERIC,
    ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_equity_curve_export (
    raw_id BIGSERIAL PRIMARY KEY,
    portfolio_id TEXT,
    timestamp TIMESTAMPTZ,
    portfolio_value NUMERIC,
    cash NUMERIC,
    drawdown NUMERIC,
    drawdown_pct NUMERIC,
    ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Validation Audit / Error Log
CREATE TABLE IF NOT EXISTS export_validation_errors (
    error_id BIGSERIAL PRIMARY KEY,
    export_type VARCHAR(32) NOT NULL,
    record_identifier TEXT,
    rule_name VARCHAR(64) NOT NULL,
    error_message TEXT NOT NULL,
    raw_payload JSONB,
    detected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 3. Production Clean Tables (strongly typed, checked, indexed)
CREATE TABLE IF NOT EXISTS clean_trades (
    trade_id TEXT PRIMARY KEY,
    symbol VARCHAR(16) NOT NULL,
    side VARCHAR(8) NOT NULL CHECK (side IN ('LONG', 'SHORT', 'BUY', 'SELL')),
    quantity NUMERIC NOT NULL CHECK (quantity > 0),
    entry_price NUMERIC NOT NULL CHECK (entry_price > 0),
    exit_price NUMERIC NOT NULL CHECK (exit_price > 0),
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ NOT NULL CHECK (exit_time >= entry_time),
    pnl NUMERIC NOT NULL,
    return_pct NUMERIC NOT NULL,
    commission NUMERIC NOT NULL DEFAULT 0.0 CHECK (commission >= 0),
    holding_period_bars INTEGER NOT NULL DEFAULT 0 CHECK (holding_period_bars >= 0),
    validated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clean_trades_symbol ON clean_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_clean_trades_times ON clean_trades(entry_time, exit_time);

CREATE TABLE IF NOT EXISTS clean_positions (
    position_id BIGSERIAL PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    name TEXT NOT NULL,
    asset_class VARCHAR(32) NOT NULL,
    sector VARCHAR(64),
    side VARCHAR(8) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity NUMERIC NOT NULL CHECK (quantity > 0),
    entry_price NUMERIC NOT NULL CHECK (entry_price > 0),
    current_price NUMERIC NOT NULL CHECK (current_price > 0),
    market_value NUMERIC NOT NULL,
    unrealized_pnl NUMERIC NOT NULL,
    unrealized_pnl_pct NUMERIC NOT NULL,
    validated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_clean_pos UNIQUE (portfolio_id, symbol, side)
);

CREATE INDEX IF NOT EXISTS idx_clean_pos_portfolio ON clean_positions(portfolio_id);

CREATE TABLE IF NOT EXISTS clean_equity_curve (
    curve_id BIGSERIAL PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    portfolio_value NUMERIC NOT NULL CHECK (portfolio_value >= 0),
    cash NUMERIC NOT NULL,
    drawdown NUMERIC NOT NULL DEFAULT 0.0 CHECK (drawdown >= 0),
    drawdown_pct NUMERIC NOT NULL DEFAULT 0.0 CHECK (drawdown_pct >= 0 AND drawdown_pct <= 1.0),
    validated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_clean_equity UNIQUE (portfolio_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_clean_equity_ts ON clean_equity_curve(portfolio_id, timestamp);
