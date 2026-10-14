-- PostgreSQL Data Cleaning and Validation Transformation Script

-- ============================================================================
-- 1. CLEAN AND VALIDATE TRADES EXPORT
-- ============================================================================

-- Log trade validation errors
INSERT INTO export_validation_errors (
    export_type,
    record_identifier,
    rule_name,
    error_message,
    raw_payload
)
SELECT
    'trades' AS export_type,
    COALESCE(trade_id, 'UNKNOWN_ID') AS record_identifier,
    CASE
        WHEN trade_id IS NULL THEN 'NOT_NULL_TRADE_ID'
        WHEN symbol IS NULL OR TRIM(symbol) = '' THEN 'NOT_NULL_SYMBOL'
        WHEN side IS NULL OR UPPER(TRIM(side)) NOT IN ('LONG', 'SHORT', 'BUY', 'SELL') THEN 'INVALID_SIDE'
        WHEN quantity IS NULL OR quantity <= 0 THEN 'POSITIVE_QUANTITY'
        WHEN entry_price IS NULL OR entry_price <= 0 THEN 'POSITIVE_ENTRY_PRICE'
        WHEN exit_price IS NULL OR exit_price <= 0 THEN 'POSITIVE_EXIT_PRICE'
        WHEN entry_time IS NULL OR exit_time IS NULL THEN 'NOT_NULL_TIMESTAMPS'
        WHEN exit_time < entry_time THEN 'TEMPORAL_ORDER_VIOLATION'
        WHEN commission IS NOT NULL AND commission < 0 THEN 'NON_NEGATIVE_COMMISSION'
        ELSE 'GENERIC_INVALID'
    END AS rule_name,
    CASE
        WHEN trade_id IS NULL THEN 'trade_id cannot be null'
        WHEN symbol IS NULL OR TRIM(symbol) = '' THEN 'symbol cannot be empty'
        WHEN side IS NULL OR UPPER(TRIM(side)) NOT IN ('LONG', 'SHORT', 'BUY', 'SELL') THEN 'side must be LONG, SHORT, BUY, or SELL'
        WHEN quantity IS NULL OR quantity <= 0 THEN 'quantity must be strictly positive'
        WHEN entry_price IS NULL OR entry_price <= 0 THEN 'entry_price must be strictly positive'
        WHEN exit_price IS NULL OR exit_price <= 0 THEN 'exit_price must be strictly positive'
        WHEN entry_time IS NULL OR exit_time IS NULL THEN 'timestamps must not be null'
        WHEN exit_time < entry_time THEN 'exit_time must be greater than or equal to entry_time'
        WHEN commission IS NOT NULL AND commission < 0 THEN 'commission cannot be negative'
        ELSE 'invalid record'
    END AS error_message,
    to_jsonb(r.*) AS raw_payload
FROM raw_trades_export r
WHERE
    trade_id IS NULL
    OR symbol IS NULL OR TRIM(symbol) = ''
    OR side IS NULL OR UPPER(TRIM(side)) NOT IN ('LONG', 'SHORT', 'BUY', 'SELL')
    OR quantity IS NULL OR quantity <= 0
    OR entry_price IS NULL OR entry_price <= 0
    OR exit_price IS NULL OR exit_price <= 0
    OR entry_time IS NULL OR exit_time IS NULL
    OR exit_time < entry_time
    OR (commission IS NOT NULL AND commission < 0);

-- Log duplicate trade_id errors
INSERT INTO export_validation_errors (
    export_type,
    record_identifier,
    rule_name,
    error_message,
    raw_payload
)
SELECT
    'trades',
    trade_id,
    'DUPLICATE_TRADE_ID',
    'duplicate trade_id detected in staging export',
    to_jsonb(r.*)
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY trade_id ORDER BY ingested_at DESC, raw_id DESC) AS row_num
    FROM raw_trades_export
    WHERE trade_id IS NOT NULL
) r
WHERE r.row_num > 1;

-- Clean and insert valid deduplicated trade records
INSERT INTO clean_trades (
    trade_id,
    symbol,
    side,
    quantity,
    entry_price,
    exit_price,
    entry_time,
    exit_time,
    pnl,
    return_pct,
    commission,
    holding_period_bars
)
SELECT
    TRIM(trade_id),
    UPPER(TRIM(symbol)),
    UPPER(TRIM(side)),
    quantity,
    entry_price,
    exit_price,
    entry_time,
    exit_time,
    COALESCE(pnl, 0.0),
    COALESCE(return_pct, 0.0),
    COALESCE(commission, 0.0),
    COALESCE(holding_period_bars, 0)
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY trade_id ORDER BY ingested_at DESC, raw_id DESC) AS row_num
    FROM raw_trades_export
    WHERE
        trade_id IS NOT NULL
        AND symbol IS NOT NULL AND TRIM(symbol) <> ''
        AND side IS NOT NULL AND UPPER(TRIM(side)) IN ('LONG', 'SHORT', 'BUY', 'SELL')
        AND quantity IS NOT NULL AND quantity > 0
        AND entry_price IS NOT NULL AND entry_price > 0
        AND exit_price IS NOT NULL AND exit_price > 0
        AND entry_time IS NOT NULL AND exit_time IS NOT NULL
        AND exit_time >= entry_time
        AND (commission IS NULL OR commission >= 0)
) valid_trades
WHERE valid_trades.row_num = 1
ON CONFLICT (trade_id) DO UPDATE SET
    symbol = EXCLUDED.symbol,
    side = EXCLUDED.side,
    quantity = EXCLUDED.quantity,
    entry_price = EXCLUDED.entry_price,
    exit_price = EXCLUDED.exit_price,
    entry_time = EXCLUDED.entry_time,
    exit_time = EXCLUDED.exit_time,
    pnl = EXCLUDED.pnl,
    return_pct = EXCLUDED.return_pct,
    commission = EXCLUDED.commission,
    holding_period_bars = EXCLUDED.holding_period_bars,
    validated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- 2. CLEAN AND VALIDATE POSITIONS EXPORT
-- ============================================================================

-- Log position validation errors
INSERT INTO export_validation_errors (
    export_type,
    record_identifier,
    rule_name,
    error_message,
    raw_payload
)
SELECT
    'positions' AS export_type,
    CONCAT(COALESCE(portfolio_id, 'PORTFOLIO'), ':', COALESCE(symbol, 'SYMBOL')) AS record_identifier,
    CASE
        WHEN symbol IS NULL OR TRIM(symbol) = '' THEN 'NOT_NULL_SYMBOL'
        WHEN portfolio_id IS NULL OR TRIM(portfolio_id) = '' THEN 'NOT_NULL_PORTFOLIO_ID'
        WHEN side IS NULL OR UPPER(TRIM(side)) NOT IN ('LONG', 'SHORT') THEN 'INVALID_POSITION_SIDE'
        WHEN quantity IS NULL OR quantity <= 0 THEN 'POSITIVE_QUANTITY'
        WHEN entry_price IS NULL OR entry_price <= 0 THEN 'POSITIVE_ENTRY_PRICE'
        WHEN current_price IS NULL OR current_price <= 0 THEN 'POSITIVE_CURRENT_PRICE'
        WHEN market_value IS NULL THEN 'NOT_NULL_MARKET_VALUE'
        ELSE 'GENERIC_INVALID'
    END AS rule_name,
    'failed position sanity check validation' AS error_message,
    to_jsonb(p.*) AS raw_payload
FROM raw_positions_export p
WHERE
    symbol IS NULL OR TRIM(symbol) = ''
    OR portfolio_id IS NULL OR TRIM(portfolio_id) = ''
    OR side IS NULL OR UPPER(TRIM(side)) NOT IN ('LONG', 'SHORT')
    OR quantity IS NULL OR quantity <= 0
    OR entry_price IS NULL OR entry_price <= 0
    OR current_price IS NULL OR current_price <= 0
    OR market_value IS NULL;

-- Clean and insert valid deduplicated position records
INSERT INTO clean_positions (
    portfolio_id,
    symbol,
    name,
    asset_class,
    sector,
    side,
    quantity,
    entry_price,
    current_price,
    market_value,
    unrealized_pnl,
    unrealized_pnl_pct
)
SELECT
    TRIM(portfolio_id),
    UPPER(TRIM(symbol)),
    COALESCE(NULLIF(TRIM(name), ''), UPPER(TRIM(symbol))),
    UPPER(COALESCE(NULLIF(TRIM(asset_class), ''), 'EQUITY')),
    NULLIF(TRIM(sector), ''),
    UPPER(TRIM(side)),
    quantity,
    entry_price,
    current_price,
    market_value,
    COALESCE(unrealized_pnl, 0.0),
    COALESCE(unrealized_pnl_pct, 0.0)
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY portfolio_id, symbol, side ORDER BY ingested_at DESC, raw_id DESC) AS row_num
    FROM raw_positions_export
    WHERE
        symbol IS NOT NULL AND TRIM(symbol) <> ''
        AND portfolio_id IS NOT NULL AND TRIM(portfolio_id) <> ''
        AND side IS NOT NULL AND UPPER(TRIM(side)) IN ('LONG', 'SHORT')
        AND quantity IS NOT NULL AND quantity > 0
        AND entry_price IS NOT NULL AND entry_price > 0
        AND current_price IS NOT NULL AND current_price > 0
        AND market_value IS NOT NULL
) valid_pos
WHERE valid_pos.row_num = 1
ON CONFLICT (portfolio_id, symbol, side) DO UPDATE SET
    name = EXCLUDED.name,
    asset_class = EXCLUDED.asset_class,
    sector = EXCLUDED.sector,
    quantity = EXCLUDED.quantity,
    entry_price = EXCLUDED.entry_price,
    current_price = EXCLUDED.current_price,
    market_value = EXCLUDED.market_value,
    unrealized_pnl = EXCLUDED.unrealized_pnl,
    unrealized_pnl_pct = EXCLUDED.unrealized_pnl_pct,
    validated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- 3. CLEAN AND VALIDATE EQUITY CURVE EXPORT
-- ============================================================================

-- Log equity curve validation errors
INSERT INTO export_validation_errors (
    export_type,
    record_identifier,
    rule_name,
    error_message,
    raw_payload
)
SELECT
    'equity_curve',
    CONCAT(COALESCE(portfolio_id, 'PORTFOLIO'), ':', COALESCE(timestamp::TEXT, 'NO_TS')),
    CASE
        WHEN portfolio_id IS NULL OR TRIM(portfolio_id) = '' THEN 'NOT_NULL_PORTFOLIO_ID'
        WHEN timestamp IS NULL THEN 'NOT_NULL_TIMESTAMP'
        WHEN portfolio_value IS NULL OR portfolio_value < 0 THEN 'NON_NEGATIVE_PORTFOLIO_VALUE'
        WHEN cash IS NULL THEN 'NOT_NULL_CASH'
        WHEN drawdown IS NOT NULL AND drawdown < 0 THEN 'NON_NEGATIVE_DRAWDOWN'
        WHEN drawdown_pct IS NOT NULL AND (drawdown_pct < 0 OR drawdown_pct > 1.0) THEN 'INVALID_DRAWDOWN_PCT'
        ELSE 'GENERIC_INVALID'
    END,
    'failed equity curve validation check',
    to_jsonb(e.*)
FROM raw_equity_curve_export e
WHERE
    portfolio_id IS NULL OR TRIM(portfolio_id) = ''
    OR timestamp IS NULL
    OR portfolio_value IS NULL OR portfolio_value < 0
    OR cash IS NULL
    OR (drawdown IS NOT NULL AND drawdown < 0)
    OR (drawdown_pct IS NOT NULL AND (drawdown_pct < 0 OR drawdown_pct > 1.0));

-- Clean and insert valid deduplicated equity curve records
INSERT INTO clean_equity_curve (
    portfolio_id,
    timestamp,
    portfolio_value,
    cash,
    drawdown,
    drawdown_pct
)
SELECT
    TRIM(portfolio_id),
    timestamp,
    portfolio_value,
    cash,
    COALESCE(drawdown, 0.0),
    COALESCE(drawdown_pct, 0.0)
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY portfolio_id, timestamp ORDER BY ingested_at DESC, raw_id DESC) AS row_num
    FROM raw_equity_curve_export
    WHERE
        portfolio_id IS NOT NULL AND TRIM(portfolio_id) <> ''
        AND timestamp IS NOT NULL
        AND portfolio_value IS NOT NULL AND portfolio_value >= 0
        AND cash IS NOT NULL
        AND (drawdown IS NULL OR drawdown >= 0)
        AND (drawdown_pct IS NULL OR (drawdown_pct >= 0 AND drawdown_pct <= 1.0))
) valid_equity
WHERE valid_equity.row_num = 1
ON CONFLICT (portfolio_id, timestamp) DO UPDATE SET
    portfolio_value = EXCLUDED.portfolio_value,
    cash = EXCLUDED.cash,
    drawdown = EXCLUDED.drawdown,
    drawdown_pct = EXCLUDED.drawdown_pct,
    validated_at = CURRENT_TIMESTAMP;
