"""Test configuration and shared fixtures."""

import numpy as np
import pytest
from click.testing import CliRunner

from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.types import (
    Asset,
    AssetClass,
    Portfolio,
    Position,
    PositionType,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def benchmark_portfolios():
    """All built-in benchmark portfolios."""
    return PortfolioRepository.get_benchmark_portfolios()


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Standard global macro sample portfolio."""
    port = PortfolioRepository.get_portfolio("global-macro")
    assert port is not None
    return port


@pytest.fixture
def simple_portfolio() -> Portfolio:
    """A minimal 2-asset portfolio for deterministic assertions."""
    asset1 = Asset(
        symbol="AAA",
        name="Asset AAA",
        asset_class=AssetClass.EQUITY,
        current_price=100.0,
        volatility_annual=0.20,
    )
    asset2 = Asset(
        symbol="BBB",
        name="Asset BBB",
        asset_class=AssetClass.EQUITY,
        current_price=50.0,
        volatility_annual=0.30,
    )
    pos1 = Position(
        asset=asset1,
        quantity=100.0,
        entry_price=90.0,
        current_price=100.0,
        side=PositionType.LONG,
    )
    pos2 = Position(
        asset=asset2,
        quantity=200.0,
        entry_price=60.0,
        current_price=50.0,
        side=PositionType.LONG,
    )
    return Portfolio(
        id="test-portfolio",
        name="Test Portfolio",
        description="Simple 2-asset portfolio for tests",
        cash=10000.0,
        positions=[pos1, pos2],
    )


@pytest.fixture
def sample_daily_returns() -> np.ndarray:
    """Deterministic synthetic return series for testing."""
    return PortfolioRepository.generate_synthetic_returns(
        n_days=500,
        mean_daily=0.0005,
        vol_daily=0.015,
        fat_tailed=True,
        seed=123,
    )
