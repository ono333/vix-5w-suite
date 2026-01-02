# Implementation Guide

## 🎯 Quick Start

### Running the App

```bash
# Install dependencies
pip install streamlit yfinance pandas numpy scipy matplotlib

# Run the app
streamlit run app.py
```

### First Run

The app will:
1. Load VIX data from Yahoo Finance (2015-present)
2. Calculate rolling 52-week percentile
3. Identify regime changes
4. Run adaptive backtest
5. Show performance by regime

**Expected result**: You should see trades executing and equity curve moving.

---

## 📁 Project Structure

```
vix_suite/
├── app.py                      # Main Streamlit app
│
├── core/
│   ├── adaptive_backtester.py  # Adaptive backtest engine
│   ├── regime_adapter.py       # Regime detection
│   ├── backtester.py           # Standard synthetic backtest
│   ├── backtester_massive.py   # For Massive API when ready
│   ├── backtester_uvxy.py      # UVXY-specific
│   ├── data_loader.py          # Data loading utilities
│   └── ...                     # Other modules
│
├── ui/
│   ├── sidebar.py              # Sidebar controls
│   └── ...                     # UI modules
│
├── experiments/
│   └── grid_scan.py            # Parameter optimization
│
├── daily_signal.py             # Standalone email script
│
└── docs/
    ├── README.md               # Main documentation
    └── TROUBLESHOOTING.md      # Debug guide
```

---

## 🔧 Customization Examples

### Example 1: Custom Regime

```python
# In core/regime_adapter.py, add to REGIME_CONFIGS:

"crash_regime": RegimeConfig(
    name="Market Crash",
    percentile_range=(0.95, 1.00),  # Top 5% of VIX
    mode="defensive",
    alloc_pct=0.0,  # Go flat during crashes
    entry_percentile=1.0,  # Never enter
    target_mult=1.0,
    exit_mult=1.0,
    long_dte_weeks=1,
    otm_pts=0.0,
    sigma_mult=1.0,
)
```

### Example 2: Add VIX Futures Filter

```python
# In core/backtester.py, modify entry logic:

# Current:
if current_vix_pct <= entry_pct_threshold:
    # Enter position

# Enhanced:
vix_futures_contango = check_contango(trade_date)  # You implement this
if (current_vix_pct <= entry_pct_threshold and 
    vix_futures_contango > 0.05):  # Only enter in contango
    # Enter position
```

### Example 3: Custom Diagnostic

```python
# In app.py, add custom analysis section:

def show_custom_analysis(bt_results):
    st.subheader("Custom Analysis")
    
    trade_log = bt_results.get("trade_log", [])
    
    # Analyze trades by day of week
    df = pd.DataFrame(trade_log)
    df["entry_dow"] = pd.to_datetime(df["entry_date"]).dt.dayofweek
    
    dow_performance = df.groupby("entry_dow")["pnl"].mean()
    
    st.bar_chart(dow_performance)
    st.caption("Average PnL by entry day of week")
```

---

## 🛠 Common Issues & Solutions

### Import Errors

**Problem:**
```python
ModuleNotFoundError: No module named 'core.regime_adapter'
```

**Solution:**
```python
# Option A: Fix import path
from core.regime_adapter import RegimeAdapter

# Option B: Add to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "core"))
from regime_adapter import RegimeAdapter
```

### Conflicting Parameter Names

**Problem:**
Your existing `params` dict has different structure than backtester expects.

**Solution:**
Create adapter function:

```python
def adapt_params(your_params):
    """Convert param structure to expected format"""
    return {
        "initial_capital": your_params["capital"],
        "risk_free": your_params["r"],
        # ... map all params ...
    }
```

---

## 🧪 Testing

### Validation Checklist

After setup, verify:

1. **App loads without errors**
```bash
streamlit run app.py
# Should see "✅ debug: main() started"
```

2. **Data loads correctly**
- Check VIX chart displays
- Verify date range matches selection

3. **Backtest executes**
- Equity curve shows movement (not flat)
- Trades > 0
- Performance metrics display

4. **Live Signals work**
- Navigate to Live Signals page
- Verify UVXY spot price loads
- Check variant signals generate

5. **Email sends**
- Configure Gmail credentials
- Test with force-send option
- Verify HTML renders correctly

---

## 📧 Email Setup

### Gmail SMTP Configuration

1. **Enable 2-Factor Authentication** on your Google account

2. **Create App Password**:
   - Go to Google Account → Security → App passwords
   - Generate password for "Mail"
   - Save the 16-character password

3. **Set Environment Variables**:
```bash
export GMAIL_USER="your.email@gmail.com"
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
```

4. **Test Email**:
```bash
# Using standalone script
python daily_signal.py --to your.email@gmail.com --force

# Or use the button in app.py Live Signals page
```

---

## 📊 Performance Expectations

### Synthetic Backtest (Black-Scholes)
- **Best case**: Strategy captures VIX mean reversion
- **Realistic case**: CAGR 10-30%, Max DD 30-50%
- **Warning**: Synthetic pricing doesn't capture:
  - Skew changes
  - Liquidity gaps
  - Bid-ask spreads
  - Early assignment risk

### With Real Historical Data
Once you have actual option chains, expect:
- Lower returns (wider spreads, worse fills)
- More realistic drawdowns
- Better sense of liquidity constraints

---

## ⚠️ Known Limitations

### 1. Synthetic Pricing
- Black-Scholes assumes constant vol (not true for VIX)
- Doesn't model skew or term structure properly
- Overestimates edge compared to real trading

### 2. Percentile-Only Entry
- Current system only looks at VIX level percentile
- Doesn't consider:
  - VIX term structure (contango/backwardation)
  - VIX futures vs spot
  - Correlation regime changes

### 3. Weekly-Only Execution
- Assumes you can trade exactly at weekly close
- Reality: need to manage intraweek moves
- Gamma risk between weekly snapshots

### 4. No Dynamic Hedging
- Positions held passively until exit rules
- Real trading might adjust delta/gamma dynamically

---

## 📜 Next Steps

### Immediate Improvements
1. ✅ **Add UVXY support** - Use `backtester_uvxy.py` for UVXY instead of VIX
2. ✅ **Grid scan** - Find optimal regime parameters
3. ✅ **Email automation** - Thursday signal delivery
4. ✅ **Transaction cost analysis** - Model real fees, slippage

### Advanced Features
1. **Machine learning regime detection**
   - Train classifier on VIX + VIX futures + market indicators
   - Predict regime changes before they fully materialize

2. **Dynamic parameter optimization**
   - Online learning: adjust parameters based on recent performance
   - Separate in-sample vs out-of-sample testing

3. **Multi-asset correlation**
   - Consider SPX level, bonds, commodities
   - Adjust VIX strategy based on broader risk-on/risk-off

4. **Real-time deployment**
   - Connect to broker API
   - Auto-execute trades based on regime signals
   - Risk management overrides

---

## ✅ Final Checklist

Before going live:

- [ ] All modules in correct directories
- [ ] Imports working without errors
- [ ] Backtest runs and shows trades
- [ ] Equity curves render correctly
- [ ] Trade logs export correctly
- [ ] Live Signals page loads
- [ ] Email sends successfully
- [ ] Documentation reviewed
- [ ] Tested on 2+ year date range

---

*Last Updated: January 2026*
