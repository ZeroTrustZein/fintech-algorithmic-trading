"""FinTech Algorithmic Trading Risk Engine & Monte Carlo VaR Analyzer."""

from fintech_algorithmic_trading.engine.evt import EVTEngine, calculate_evt_var
from fintech_algorithmic_trading.engine.greeks import calculate_option_greeks
from fintech_algorithmic_trading.engine.hedging import OptionHedgingEngine
from fintech_algorithmic_trading.engine.monte_carlo import MonteCarloEngine
from fintech_algorithmic_trading.engine.optimizer import PortfolioOptimizer
from fintech_algorithmic_trading.engine.risk_engine import PortfolioRiskEngine
from fintech_algorithmic_trading.engine.sizing import PositionSizer
from fintech_algorithmic_trading.engine.var import (
    VaRCalculator,
    calculate_cornish_fisher_var,
    calculate_historical_var,
    calculate_parametric_var,
)
from fintech_algorithmic_trading.storage.repository import PortfolioRepository
from fintech_algorithmic_trading.types import (
    Asset,
    AssetClass,
    DeltaGammaHedge,
    DriftModel,
    EVTResult,
    KellyCriterionMode,
    KellySizingResult,
    MonteCarloConfig,
    MonteCarloResult,
    OptimizationObjective,
    OptimizationResult,
    OptionContract,
    OptionGreeks,
    OptionType,
    Portfolio,
    PortfolioGreeks,
    Position,
    PositionType,
    RiskComplianceReport,
    RiskLimits,
    RiskViolation,
    StressScenario,
    StressTestResult,
    VaRMethod,
    VaRResult,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Asset",
    "AssetClass",
    "DeltaGammaHedge",
    "DriftModel",
    "EVTEngine",
    "EVTResult",
    "KellyCriterionMode",
    "KellySizingResult",
    "MonteCarloConfig",
    "MonteCarloEngine",
    "MonteCarloResult",
    "OptionContract",
    "OptionGreeks",
    "OptionHedgingEngine",
    "OptionType",
    "OptimizationObjective",
    "OptimizationResult",
    "Portfolio",
    "PortfolioGreeks",
    "PortfolioOptimizer",
    "PortfolioRepository",
    "PortfolioRiskEngine",
    "Position",
    "PositionSizer",
    "PositionType",
    "RiskComplianceReport",
    "RiskLimits",
    "RiskViolation",
    "StressScenario",
    "StressTestResult",
    "VaRCalculator",
    "VaRMethod",
    "VaRResult",
    "calculate_cornish_fisher_var",
    "calculate_evt_var",
    "calculate_historical_var",
    "calculate_option_greeks",
    "calculate_parametric_var",
]
