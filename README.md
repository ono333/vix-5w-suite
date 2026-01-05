# VIX 5% Weekly Suite - Improved Version

## What's New

### 1. **Trade Logger for 5 Paper Trade Variants** 📊

The new `trade_logger.py` module supports tracking trades from 5 different strategy variants:

| Variant | Name | Description |
|---------|------|-------------|
| **V1** | Static Baseline | Fixed parameters, no adaptation |
| **V2** | Regime Adaptive | Adjusts params based on VIX percentile |
| **V3** | Aggressive Entry | Higher entry threshold, larger positions |
| **V4** | Conservative | Lower entry threshold, smaller positions |
| **V5** | High VIX Contrarian | Enter when VIX is elevated |

Each trade entry captures:
- Variant ID and name
- Entry/exit timestamps and prices
- Position details (strikes, DTE, contracts)
- P&L metrics
- Regime at entry/exit
- Strategy parameters used

```python
from trade_logger import TradeLogger, VariantConfig

# Initialize logger with persistence
logger = TradeLogger(log_path="paper_trades.json")

# Log a trade entry for Variant 1
trade = logger.log_entry(
    variant_id=1,
    entry_date="2024-01-15",
    entry_underlying=15.5,
    contracts=10,
    long_strike=20.0,
    long_dte_weeks=26,
    entry_cost=3500.0,
    entry_percentile=0.22,
    entry_threshold=0.25,
)

# Log exit when trade closes
logger.log_exit(
    trade_id=trade.trade_id,
    exit_date="2024-02-01",
    exit_underlying=18.2,
    exit_value=4200.0,
    exit_reason="target",
)

# Compare all variants
stats_df = logger.get_all_variants_stats()
print(stats_df)

# Export to Excel with per-variant sheets
logger.export_excel("paper_trade_comparison.xlsx")
```

### 2. **Trade Log Display** 📊
The Backtester page now shows a complete trade log with:
- Entry/Exit dates
- Duration (weeks)
- Entry/Exit equity
- PnL ($ and %)
- Strike prices
- Entry/Exit regimes (in adaptive mode)

### 2. **Trade Statistics**
Key metrics displayed prominently:
- Total Trades
- Win Rate
- Average Duration
- Sharpe Ratio

### 3. **Regime-Adaptive Strategy** 🎯
Enable "Regime-Adaptive Mode" in the sidebar to automatically adjust parameters based on VIX percentile:

| Regime | VIX Percentile | Entry % | DTE | Position Size | Behavior |
|--------|---------------|---------|-----|---------------|----------|
| Ultra Low | 0-10% | 0.35 | 26w | 1.5% | Aggressive, expect vol to stay low |
| Low | 10-25% | 0.30 | 26w | 1.2% | Normal calm market params |
| Medium | 25-50% | 0.25 | 15w | 1.0% | Balanced approach |
| High | 50-75% | 0.20 | 8w | 0.8% | Defensive, smaller positions |
| Extreme | 75-100% | 0.15 | 5w | 0.5% | Very defensive, quick exits |

### 4. **Trade Explorer Page** 🔍
Now fully functional with:
- Complete trade log table
- PnL distribution chart
- Duration analysis
- Trades marked on price chart
- CSV export

### 5. **Regime Analysis Page** 📈
New page showing:
- Time spent in each regime
- Trade performance by entry regime
- Regime timeline visualization

## Installation

1. **Copy the new files to your project:**
```bash
# From the downloaded folder:
cp app_improved.py /path/to/01_vix_5w_suite/app.py
cp sidebar_improved.py /path/to/01_vix_5w_suite/ui/sidebar.py
cp regime_adapter.py /path/to/01_vix_5w_suite/core/regime_adapter.py
```

2. **Or rename and keep both versions:**
```bash
mv app.py app_original.py
mv app_improved.py app.py
```

## Why Your Backtest Shows 0% Returns

Looking at your screenshot, the issue is:

1. **Massive Historical has no data** - The Massive API doesn't provide historical option chain snapshots via REST. Every chain query returns empty, so no trades execute.

2. **Solution**: Switch to "Synthetic (BS)" pricing source in the sidebar. This uses Black-Scholes pricing which always works.

3. **Entry percentile may be too restrictive** - If set to 0.10 (10th percentile), you'll only enter when VIX is very low. Try 0.25-0.35 for more trades.

## Quick Start

1. Run the app:
```bash
cd 01_vix_5w_suite
streamlit run app.py
```

2. In the sidebar:
   - Set **Pricing source** to "Synthetic (BS)"
   - Set **Entry percentile** to 0.25-0.35
   - Enable **Regime-Adaptive Mode** (optional)

3. Navigate to **Backtester** page to see trades and the trade log

## Files Included

| File | Description |
|------|-------------|
| `app_improved.py` | Main app with trade log display |
| `sidebar_improved.py` | Updated sidebar with regime toggle |
| `regime_adapter.py` | Regime-adaptive strategy logic (fixed) |
| `trade_logger.py` | **NEW** - Multi-variant paper trade logger |

## Installation for 5-Variant Paper Trading

1. **Copy files to your project:**
```bash
cp app_improved.py /path/to/01_vix_5w_suite/app.py
cp sidebar_improved.py /path/to/01_vix_5w_suite/ui/sidebar.py
cp regime_adapter.py /path/to/01_vix_5w_suite/core/regime_adapter.py
cp trade_logger.py /path/to/01_vix_5w_suite/core/trade_logger.py
```

2. **Run backtests and log to trade logger:**
```python
from trade_logger import TradeLogger, create_trade_log_from_backtest

# Run backtest for each variant
logger = TradeLogger(log_path="paper_trades.json")

# Variant 1: Static baseline
bt1 = run_backtest(vix_weekly, params_v1)
create_trade_log_from_backtest(bt1, variant_id=1, vix_weekly, params_v1, logger)

# Variant 2: Regime adaptive  
bt2 = run_regime_adaptive_backtest(vix_weekly, params_v2)
create_trade_log_from_backtest(bt2, variant_id=2, vix_weekly, params_v2, logger)

# ... repeat for V3, V4, V5

# Export comparison
logger.export_excel("variant_comparison.xlsx")
```

## Regime Parameter Optimization

The default regime parameters are based on general volatility trading principles. For best results:

1. Run the **Grid Scan** with regime-adaptive mode OFF to find optimal static parameters
2. Use those results to customize the regime configs in `regime_adapter.py`
3. Enable regime-adaptive mode for live trading

## Key Insights from Your Project History

Based on your param_history.json:
- Entry percentile around 0.90 (90th percentile - HIGH VIX) with sigma_mult=1.0 and otm_pts=5 showed good results
- This suggests entering when VIX is elevated (contrarian approach)
- Consider adjusting the regime adapter to reflect this insight
