# VIX 5% Weekly Suite - Unified Application

A comprehensive Streamlit application combining historical backtesting with regime-adaptive paper trading for UVXY volatility strategies.

## Two Operational Modes

### 📊 Research Mode (Historical Analysis)
- **Dashboard**: Equity curves, VIX/UVXY percentile visualization
- **Backtester**: Grid scan, parameter optimization, XLSX export
- **Trade Explorer**: Historical trade analysis

### 📈 Paper Trading Mode (Live Signals)
- **Signal Dashboard**: Thursday signal generation with 5 strategy variants
- **Execution Window**: Friday-Monday execution tracking
- **Active Trades**: Open position management
- **Post-Mortem Review**: Exit classification and lessons learned
- **Variant Analytics**: Promotion decisions, operational metrics
- **System Health**: Status monitoring

## 5 Strategy Variants (Role-Based)

| Variant | Role | Description | Active Regimes |
|---------|------|-------------|----------------|
| V1 | Income Harvester | Stability anchor, frequent small gains | CALM, DECLINING |
| V2 | Mean Reversion Accelerator | Post-spike decay capture | DECLINING |
| V3 | Shock Absorber | Crisis hedge, drawdown reduction | STRESSED, EXTREME |
| V4 | Convex Tail Hunter | Rare explosive payoffs | EXTREME |
| V5 | Regime-Aware Allocator | Meta-controller, sizing | ALL |

## Volatility Regimes

| Regime | VIX Percentile | Description |
|--------|---------------|-------------|
| CALM | 0-25% | Low vol, stable. Income strategies. |
| RISING | 25-50% | Vol increasing. Reduce exposure. |
| STRESSED | 50-75% | High vol. Focus on hedges. |
| DECLINING | 75-90% | Post-spike decay. Mean reversion. |
| EXTREME | 90-100% | Tail event. Maximize convexity. |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py

# 3. Select mode in sidebar:
#    - Research: Historical backtesting
#    - Paper Trading: Live signal generation
```

## Directory Structure

```
vix_suite/
├── app.py                    # Unified Streamlit application
├── requirements.txt          # Python dependencies
│
├── core/                     # Backtesting engine
│   ├── backtester.py        # Synthetic (Black-Scholes) engine
│   ├── backtester_massive.py # Massive API engine
│   ├── data_loader.py       # Data loading utilities
│   ├── param_history.py     # Parameter history tracking
│   └── ...
│
├── experiments/              # Research tools
│   ├── grid_scan.py         # Parameter optimization
│   └── ...
│
├── regime_detector.py        # Regime classification
├── variant_generator.py      # Signal generation
├── robustness_scorer.py      # Execution survivability scoring
├── trade_log.py             # Trade logging with leg tracking
├── exit_detector.py         # Exit signal detection
└── notification_engine.py   # Email notifications
```

## Workflow

### Thursday 4:30 PM
1. Open Signal Dashboard
2. Review current regime
3. Generate signal batch
4. Freeze batch when ready

### Friday-Monday (Execution Window)
1. Open Execution Window
2. Review active variants + robustness scores
3. Log paper trades with actual entry prices

### Ongoing
1. Monitor Active Trades
2. Update prices as needed
3. Close positions with documented reasoning
4. Review in Post-Mortem
5. Track metrics in Variant Analytics

## Storage

Data is stored in `~/.vix_suite/`:
- `trade_log.json` - All paper trades
- `current_signal_batch.json` - Active signals
- `regime_history.json` - Regime transition history

## Configuration

For email notifications, set environment variables:
```bash
export VIX_SMTP_HOST="smtp.gmail.com"
export VIX_SMTP_PORT="587"
export VIX_SMTP_USER="your@email.com"
export VIX_SMTP_PASS="your-app-password"
export VIX_EMAIL_TO="your@email.com"
```

---
Built for LBR-grade paper trading research 📈
