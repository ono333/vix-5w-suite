# VIX 5% Weekly Suite — Position-Aware Signal Generation

A sophisticated Streamlit application for VIX/UVXY volatility trading using diagonal spreads with **position-aware signal generation**.

## 🎯 Key Enhancement: Position-Aware Signals

The Signal Dashboard now intelligently distinguishes between:

### 🔵 OPEN POSITIONS — Management Mode
When you already hold a position for a variant:
- Shows P&L, DTE, health status
- Displays management actions (Roll, Expire, Close)
- Provides roll recommendations based on short leg DTE
- **Does NOT recommend new entries**

### ✅ ENTRY CANDIDATES — Active in Current Regime
When you have NO position and the variant is active:
- Shows entry signal with strikes, credits, targets
- Displays robustness score
- Checks entry percentile condition
- **Recommends entry if conditions are met**

### ⛔ INACTIVE VARIANTS — Not Active in Current Regime
When the variant is not suitable for current market conditions:
- Shows which regimes the variant is designed for
- Parameters visible but collapsed
- **No action recommended**

## 📁 Project Structure

```
vix_5w_suite_enhanced/
├── app.py                  # Main Streamlit application (3488 lines)
├── enums.py                # Volatility regimes, variant roles, trade status
├── regime_detector.py      # Market regime classification
├── variant_generator.py    # V1-V5 strategy variant definitions
├── trade_log.py            # Position tracking with roll management
├── robustness_scorer.py    # Signal quality scoring
├── exit_detector.py        # Exit condition detection
├── notification_engine.py  # Email/Slack notifications
├── requirements.txt        # Python dependencies
├── utils/
│   ├── __init__.py
│   └── regime_utils.py     # Regime helper functions
└── README.md
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📊 Strategy Variants (V1-V5)

| Variant | Role | Active Regimes | Entry Percentile |
|---------|------|----------------|------------------|
| V1 | Income Harvester | CALM, DECLINING | ≤25% |
| V2 | Mean Reversion | DECLINING | ≤60% |
| V3 | Shock Absorber | RISING, STRESSED, EXTREME | ≥75% |
| V4 | Tail Hunter | EXTREME | ≥90% |
| V5 | Regime Allocator | ALL (adaptive) | Varies |

## 🔄 Position Management Features

### Health Status System
- 🟢 **HEALTHY**: No immediate action needed
- 🟡 **ATTENTION**: Action recommended (short DTE < 7 days)
- 🔴 **CRITICAL**: Immediate action required (short DTE ≤ 3 days)

### Roll Tracking
- Tracks all short leg rolls with credits/debits
- Calculates cumulative roll P&L
- Suggests optimal roll timing

### Commission Tracking
- Entry commissions (long + short)
- Roll commissions
- Total cost basis calculation

## 📈 Application Modes

### Research Mode
- **Dashboard**: Equity curves, VIX percentile visualization
- **Backtester**: Grid scan, parameter optimization
- **Trade Explorer**: Historical trade analysis

### Paper Trading Mode
- **Signal Dashboard**: Position-aware signal generation ⭐
- **Execution Window**: Friday-Monday execution tracking
- **Active Trades**: Open position management
- **Trade Log**: Complete trade history
- **Variant Analytics**: Performance by variant
- **System Health**: Module status checks

## 🔒 Position-Aware Logic

```python
# Core logic in render_signal_dashboard()
existing_positions = {}  # Maps V1, V2, etc. to positions

for variant in batch.variants:
    prefix = variant.variant_id.split("-")[0]  # "V1", "V2", etc.
    has_position = prefix in existing_positions
    
    if has_position:
        # Show MANAGEMENT MODE
        # Roll, expire, close actions
    elif is_active:
        # Show ENTRY SIGNAL
        # Strike, credit, targets
    else:
        # Show INACTIVE
        # Parameters only
```

## ⚠️ Important Notes

1. **One Position Per Variant**: Each variant (V1-V5) can have at most ONE open position
2. **Diagonal Spreads**: Long LEAP call + Short weekly call
3. **Roll Before Expiry**: Short legs should be rolled when DTE ≤ 3 days
4. **Percentile Entry**: Enter when UVXY percentile ≤ variant threshold

---

*Built with Streamlit • Position-aware enhancement for realistic trading simulation*
