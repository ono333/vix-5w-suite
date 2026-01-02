# VIX 5% Weekly Suite — Regime-Adaptive Version

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

---

## 🚀 Getting Started

### Installation

```bash
# Clone/copy the project
cd vix_suite

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Quick Start Workflow

1. **Dashboard**: View current market regime and historical distribution
2. **Per-Regime Optimizer**: Run optimization for each regime (do this first!)
3. **Adaptive Backtester**: Run backtest with regime-adaptive parameters
4. **Live Signals**: Generate real-time trading recommendations

---

## 📁 Project Structure

```
vix_suite/
├── app.py                   # Main Streamlit app
├── daily_signal.py          # Standalone email script
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

---

## 📊 Pages

### Dashboard
- Current market regime detection
- 52-week percentile visualization
- Regime timeline and distribution

### Backtester
- Standard backtest with fixed parameters
- Grid scan for parameter optimization
- Equity curves and PnL breakdown
- Results export to XLSX

### Live Signals
- Real-time UVXY signal generation
- Multiple trade variants (different DTEs)
- Email automation for Thursday signals
- Position sizing recommendations

### Trade Explorer
- Detailed trade analysis
- Entry/exit visualization

---

## 📈 Strategy Logic

### Entry Rules

For each week:
1. Calculate current VIX percentile vs last 52 weeks
2. Determine regime (ultra_low/low/medium/high/extreme)
3. Check if VIX percentile <= regime's `entry_percentile` threshold
4. If yes, open position using regime's parameters

### Position Structure

#### Diagonal Spread (Low/Mid VIX)
- **Long**: 26-week call, OTM by X points
- **Short**: 1-week call, OTM by X points
- Short rolled weekly until long expires or exit triggered

#### Long-Only (High VIX)
- **Long**: 8-week call only
- No short leg (avoid negative gamma when VIX elevated)

### Exit Rules

1. **Profit target**: Long call value >= entry_cost × target_mult
2. **Stop loss**: Long call value <= entry_cost × exit_mult
3. **Expiration**: Long call DTE = 0

---

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
| `realism` | P&L haircut (1.0 = none, 0.8 = 20%) | 0.5 - 1.0 |

---

## ⚙️ Regime Configuration

Edit `REGIME_CONFIGS` in `core/regime_adapter.py`:

```python
"low_vix": RegimeConfig(
    name="Low VIX Regime",
    percentile_range=(0.10, 0.25),
    mode="diagonal",
    alloc_pct=0.01,
    entry_percentile=0.15,
    target_mult=1.30,
    exit_mult=0.50,
    long_dte_weeks=26,
    otm_pts=10.0,
    sigma_mult=0.8,
),
```

---

## 📧 Email Automation

Thursday signal emails with compact HTML format:
- Market state (VIX, UVXY, percentile, regime)
- Multiple trade variants with bid/ask data
- Position sizing recommendations

### Setup Gmail SMTP
```bash
export GMAIL_USER="your.email@gmail.com"
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
```

### Send via script
```bash
python daily_signal.py --to your.email@gmail.com
python daily_signal.py --force  # Send even if no signal
```

---

## ⚠️ Known Limitations

1. **Synthetic Pricing** - Black-Scholes doesn't capture skew or liquidity
2. **Weekly Execution** - Assumes trades at weekly close only
3. **No Dynamic Hedging** - Positions held passively until exit
4. **Percentile-Only Entry** - Doesn't consider VIX term structure

---

## 📊 Performance Expectations

### Synthetic Backtest
- Realistic case: CAGR 10-30%, Max DD 30-50%
- Results are optimistic vs real trading

### With Real Data
- Lower returns (wider spreads, worse fills)
- More realistic drawdowns

---

## 📄 License

MIT License - Free to use, modify, distribute.

**Disclaimer**: Educational purposes only. Not financial advice. Options trading involves substantial risk.

---

*"Different market conditions demand different strategies. This system adapts automatically."*
