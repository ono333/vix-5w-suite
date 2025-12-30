# VIX 5% Weekly Suite – Regime-Adaptive Version

A sophisticated VIX/UVXY options backtesting system with **adaptive parameter switching** based on market volatility regimes.

## 🌟 Key Features

### Regime-Based Adaptive Trading
The system automatically detects five distinct market volatility regimes and applies optimized parameters for each:

| Regime | VIX Percentile | Description | Strategy Stance |
|--------|---------------|-------------|-----------------|
| **Ultra Low** | 0-10% | Extremely calm markets | Aggressive entry, higher allocation |
| **Low** | 10-25% | Primary entry zone | Standard positioning |
| **Medium** | 25-50% | Normal volatility | Moderate, selective entries |
| **High** | 50-75% | Elevated volatility | Defensive, long-only mode |
| **Extreme** | 75-100% | Crisis/spike | Avoid new positions |

### Per-Regime Optimization
Instead of using one-size-fits-all parameters, the system:
1. Separates historical data by regime
2. Runs independent grid scans for each regime's historical periods
3. Stores optimized parameters per regime
4. Dynamically applies the right parameters based on current conditions

## 📁 Project Structure

```
vix_suite/
├── app_improved.py          # Main Streamlit app with regime features
├── requirements.txt         # Python dependencies
│
├── core/                    # Core backtesting modules
│   ├── adaptive_backtester.py   # Regime-adaptive backtest engine
│   ├── backtester.py           # Standard synthetic backtest
│   ├── regime_adapter.py       # Regime detection & configuration
│   ├── param_history.py        # Parameter storage & retrieval
│   ├── data_loader.py          # VIX/UVXY data loading
│   ├── indicators.py           # Technical indicators
│   ├── metrics.py              # Performance calculations
│   └── math_engine.py          # Black-Scholes pricing
│
├── experiments/             # Grid scanning & optimization
│   ├── grid_scan.py            # Standard parameter grid scan
│   ├── per_regime_grid_scan.py # Per-regime optimization
│   ├── entry_rules.py          # Entry signal logic
│   └── exit_rules.py           # Exit signal logic
│
├── ui/                      # User interface components
│   ├── sidebar.py              # Sidebar controls
│   ├── charts.py               # Chart rendering
│   ├── tables.py               # Table displays
│   └── styles.py               # CSS styling
│
└── config/                  # Configuration
    ├── defaults.py             # Default parameters
    └── massive_config.py       # Massive API config (if used)
```

## 🚀 Getting Started

### Installation

```bash
# Clone/copy the project
cd vix_suite

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app_improved.py
```

### Quick Start Workflow

1. **Dashboard**: View current market regime and historical distribution
2. **Per-Regime Optimizer**: Run optimization for each regime (do this first!)
3. **Adaptive Backtester**: Run backtest with regime-adaptive parameters
4. **Compare**: See how adaptive beats static parameter approaches

## 📊 Pages

### Dashboard
- Current market regime detection
- 52-week percentile visualization
- Regime timeline and distribution
- Stored optimized parameters overview

### Backtester
- Standard backtest with fixed parameters
- Grid scan for parameter optimization
- Equity curves and PnL breakdown
- Results export to XLSX

### Adaptive Backtester
- **New!** Regime-aware parameter switching
- Uses optimized params from Per-Regime Optimizer
- Compares adaptive vs static performance
- Per-regime trade statistics
- Regime transition tracking

### Per-Regime Optimizer
- **New!** Separate grid scans per regime
- Optimizes entry %, OTM distance, sigma, DTE
- Saves best params to history
- Shows comparison table across regimes

### Trade Explorer
- Detailed trade analysis (coming soon)
- Entry/exit visualization
- Regime-tagged performance

## ⚙️ Regime Configuration

Default regime configurations in `core/regime_adapter.py`:

```python
REGIME_CONFIGS = {
    "ULTRA_LOW": RegimeConfig(
        percentile_range=(0.0, 0.10),
        entry_percentile=0.08,
        alloc_pct=0.015,
        otm_pts=8.0,
        target_mult=1.30,
    ),
    "LOW": RegimeConfig(
        percentile_range=(0.10, 0.25),
        entry_percentile=0.15,
        alloc_pct=0.01,
        otm_pts=10.0,
    ),
    # ... etc
}
```

## 🔧 Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `entry_percentile` | VIX percentile threshold for entry | 0.05 - 0.30 |
| `otm_pts` | OTM distance in VIX points | 5 - 20 |
| `sigma_mult` | Volatility multiplier for pricing | 0.5 - 2.0 |
| `long_dte_weeks` | LEAP option duration | 8 - 52 |
| `alloc_pct` | Equity allocation per trade | 0.005 - 0.02 |
| `target_mult` | Profit target multiplier | 1.10 - 1.50 |
| `exit_mult` | Stop loss multiplier | 0.40 - 0.70 |

## 📈 Strategy Logic

### Diagonal Spread (Default)
- **Long leg**: OTM LEAP call (13-26 weeks)
- **Short leg**: Weekly OTM call (rolled weekly)
- Premium collection from short leg offsets time decay

### Long Only (High Vol Regimes)
- Simple long call position
- No short leg exposure
- Used when VIX is elevated (less room for weekly premium)

## 🎯 Optimization Criteria

- **Balanced**: Equal weight to CAGR and max drawdown (recommended)
- **CAGR**: Pure return maximization
- **MaxDD**: Risk minimization (lowest drawdown)

## 📝 Notes

### Data Sources
- **Yahoo Finance**: VIX and UVXY historical data
- **Massive API**: Optional for real option chains (requires API key)

### Realism Features
- Transaction fees ($0.65/contract default)
- Slippage modeling (5 bps default)
- Position size limits (10,000 contracts max)
- Realism haircut multiplier

### Risk Controls
- Maximum position sizing (% of equity)
- Regime-based allocation adjustment
- Mode switching (diagonal → long-only) in high vol

## 🔄 Version History

- **v2.0** - Regime-adaptive system with per-regime optimization
- **v1.0** - Original static parameter backtester

## 📧 Support

Built for the VIX 5% Weekly strategy research project.

For issues or questions, check the code comments or open an issue.

---

*"Different market conditions demand different strategies. This system adapts automatically."*
