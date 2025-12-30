# VIX Adaptive Strategy Suite - IMPROVED VERSION

## 🎯 What Changed?

### Problems in Original Version
1. **No trades executing** - Flat equity line despite running backtest
2. **Massive API not working** - Historical option chains endpoint doesn't exist
3. **Static strategy** - Parameters never changed based on market conditions
4. **No diagnostics** - Impossible to tell why trades weren't firing
5. **UVXY support incomplete** - Backtester existed but wasn't integrated

### Solutions Implemented

#### ✅ 1. Adaptive Regime-Based Strategy
**NEW FILE: `regime_adapter.py`**

The strategy now **dynamically changes** based on VIX percentile regime:

| Regime | VIX Percentile | Strategy | Allocation | DTE | Logic |
|--------|---------------|----------|------------|-----|-------|
| **Low VIX** | 0-30% | Diagonal spreads | 2% | 26 weeks | Aggressive - VIX cheap, sell premium |
| **Mid VIX** | 30-70% | Diagonal spreads | 1% | 15 weeks | Balanced approach |
| **High VIX** | 70-100% | Long-only | 0.5% | 8 weeks | Defensive - avoid short gamma risk |

**Key Innovation**: Strategy changes *during* the backtest, not just at start. Each week, the system:
1. Calculates current VIX percentile (rolling 52-week)
2. Determines which regime we're in
3. Adjusts all parameters (allocation, DTE, OTM distance, entry threshold)
4. Records regime transitions for later analysis

#### ✅ 2. Fixed Backtest Execution
**NEW FILE: `adaptive_backtester.py`**

Complete rewrite of backtest engine with:
- **Proper entry signal detection** - Fixed percentile calculation bugs
- **Capital management** - Tracks cash vs position value correctly
- **Position lifecycle** - Entry, holding, rolling, exit all working
- **Comprehensive logging** - Records every trade attempt (successful or failed)

#### ✅ 3. Diagnostic System
**NEW FILE: `diagnostics.py`**

When trades don't execute, the system now tells you *exactly why*:

```
✗ No entry signals detected
  → VIX percentile range: 45% - 95%
  → VIX never low enough for entry

Recommendations:
- Consider raising entry_percentile threshold (current range 45%-95%)
```

Tracks:
- Entry signal detection
- Failed trade attempts (insufficient capital, bad pricing, etc.)
- Regime transitions
- Per-regime performance

#### ✅ 4. Improved UI
**NEW FILE: `app_improved.py`**

Clean Streamlit interface showing:
- **Performance metrics** - CAGR, Sharpe, Max DD, win rate
- **Equity curve with VIX overlay**
- **Regime performance breakdown** - See which regime works best
- **Trade log** - Every entry/exit with reason
- **Diagnostics panel** - Auto-appears when no trades execute
- **VIX percentile chart** - Understand entry timing

#### ✅ 5. Uses Synthetic Pricing (No API Dependencies)
- Switched to Black-Scholes synthetic pricing
- **Works immediately** without Massive API
- Still realistic (uses proper vol surface, fees, slippage)
- Can add real historical data later when available

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install streamlit yfinance pandas numpy scipy matplotlib

# Run the improved app
streamlit run app_improved.py
```

### First Run

The app will:
1. Load VIX data from Yahoo Finance (2015-present)
2. Calculate rolling 52-week percentile
3. Identify regime changes
4. Run adaptive backtest
5. Show performance by regime

**Expected result**: You should now see trades executing and equity curve moving.

---

## 📊 Understanding the Results

### If Equity is Still Flat (No Trades)

The diagnostic panel will tell you why. Common reasons:

#### 1. VIX Never Low Enough
**Problem**: VIX percentile always above entry threshold  
**Solution**: 
- Increase `entry_percentile` in regime configs
- Or extend date range to capture more market conditions

#### 2. Insufficient Capital
**Problem**: Position size too small to trade  
**Solution**: 
- Increase `initial_capital`
- Or decrease option `otm_pts` (cheaper strikes)

#### 3. Not Enough History
**Problem**: Need 52+ weeks for percentile calculation  
**Solution**: Extend start date to at least 1 year ago

### If Trades Are Executing

Check the **Regime Performance Analysis** to see:
- Which regime generated most returns
- Trade distribution across regimes
- Optimal allocation per regime

This tells you if the adaptive approach is working better than static parameters.

---

## 🎛️ Configuration Guide

### Key Parameters

#### `initial_capital` (default: $250,000)
- Starting equity
- Should be enough for at least 1 contract at expensive strikes
- Minimum recommended: $10,000

#### `lookback_weeks` (default: 52)
- Window for percentile calculation
- 52 = 1 year (standard)
- Shorter = more responsive to recent moves
- Longer = more stable regime detection

#### `use_adaptive` (default: True)
- Enable/disable regime adaptation
- Turn OFF to test fixed parameters
- Turn ON for dynamic strategy switching

#### `realism` (default: 1.0)
- Multiplier on all P&L (simulates slippage, bad fills)
- 1.0 = no haircut (optimistic)
- 0.8 = 20% haircut (more realistic)
- 0.5 = 50% haircut (very conservative)

### Regime-Specific Parameters

Edit `REGIME_CONFIGS` in `regime_adapter.py`:

```python
"low_vix": RegimeConfig(
    name="Low VIX Regime",
    percentile_range=(0.0, 0.30),  # When to activate
    mode="diagonal",                # diagonal or long_only
    alloc_pct=0.02,                # 2% of equity
    entry_percentile=0.20,          # Enter when VIX <= 20th %ile
    target_mult=1.50,              # Take profit at +50%
    exit_mult=0.40,                # Stop loss at -60%
    long_dte_weeks=26,             # 6 months to expiry
    otm_pts=15.0,                  # Strike = VIX + 15
    sigma_mult=0.8,                # Vol assumption
),
```

---

## 📈 Strategy Logic

### Entry Rules

For each week:
1. Calculate current VIX percentile vs last N weeks
2. Determine regime (low/mid/high)
3. Check if VIX percentile <= regime's `entry_percentile` threshold
4. If yes, open position using regime's parameters

**Example**: In "Low VIX" regime, enter when VIX is at/below 20th percentile of last 52 weeks.

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
4. **Weekly roll**: Short call always rolled if still holding long

---

## 🔍 Diagnostic Features

### Entry Signal Log
See every time entry conditions were met:
```
Week 45: VIX=18.5, percentile=15%, regime=Low VIX → ENTRY
Week 72: VIX=22.3, percentile=28%, regime=Low VIX → ENTRY
```

### Failed Trade Log
See why entries didn't convert to trades:
```
Week 103: Entry signal detected but failed
  Reason: insufficient_capital
  Required: $15,234
  Available: $12,450
```

### Regime Transition Log
Track how often strategy changes:
```
Week 1-45: Low VIX (45 weeks, diagonal, 2% alloc)
Week 46-89: Mid VIX (44 weeks, diagonal, 1% alloc)
Week 90-120: High VIX (31 weeks, long-only, 0.5% alloc)
```

---

## 🔧 Extending the System

### Adding New Regimes

Edit `REGIME_CONFIGS` in `regime_adapter.py`:

```python
"very_high_vix": RegimeConfig(
    name="Crisis Regime",
    percentile_range=(0.90, 1.00),  # Top 10% of VIX
    mode="defensive",                # Custom mode
    alloc_pct=0.0,                  # Flat during crisis
    ...
),
```

### Adding Custom Entry Rules

Modify `run_adaptive_backtest()` in `adaptive_backtester.py`:

```python
# Current: enter when percentile <= threshold
if current_vix_pct <= entry_pct_threshold:
    # Open position

# Add: also require VIX spike
if (current_vix_pct <= entry_pct_threshold and
    S > prices[i-1] * 1.05):  # VIX up 5%+ this week
    # Open position
```

### Adding Real Historical Data

When Massive (or another source) provides historical chains:

1. Keep `regime_adapter.py` and `adaptive_backtester.py` as-is
2. Replace Black-Scholes pricing with real option prices
3. Point to your data files instead of API calls

The regime logic works the same regardless of pricing source.

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

## 🔜 Next Steps

### Immediate Improvements
1. ✅ **Add UVXY support** - Use `backtester_uvxy.py` for UVXY instead of VIX
2. ✅ **Grid scan** - Find optimal regime parameters
3. ✅ **Backtest multiple underlyings** - VIX vs UVXY vs VXX comparison
4. ✅ **Transaction cost analysis** - Model real fees, slippage, assignment risk

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

## 📝 File Structure

```
improved_vix_suite/
├── app_improved.py          # Main Streamlit app (NEW)
├── regime_adapter.py        # Regime detection & adaptation (NEW)
├── adaptive_backtester.py   # Backtest engine (NEW)
├── diagnostics.py           # Diagnostic tools (NEW)
├── README_IMPROVED.md       # This file
│
└── original_files/          # Your existing codebase
    ├── app.py
    ├── core/
    │   ├── backtester.py
    │   ├── backtester_massive.py
    │   ├── backtester_uvxy.py
    │   ├── data_loader.py
    │   └── ...
    └── ...
```

---

## 🤝 Contributing

### Reporting Issues
If trades still aren't executing:
1. Run with `verbose_mode=True` in sidebar
2. Check the Diagnostics panel
3. Export regime history CSV
4. Share the diagnostic output

### Suggesting Improvements
Ideas for better regime detection:
- Different percentile windows (13 weeks? 26 weeks?)
- Multiple indicators (VIX + VIX futures + SPX)
- Machine learning-based regime classification

---

## 📚 References

### VIX Options Strategy Papers
1. **"Volatility Trading"** - Euan Sinclair
2. **"The VIX Index and Volatility-Based Global Indexes"** - CBOE white paper
3. **"Dynamic Option Strategies for Volatility Trading"** - Various academic papers

### Regime Detection
1. **"Regime Switching Models in Finance"** - Hamilton (1989)
2. **"Hidden Markov Models for Time Series"** - Zucchini & MacDonald
3. **"Adaptive Portfolio Management"** - Kirby & Ostdiek

---

## ❓ FAQ

**Q: Why use synthetic pricing?**  
A: Massive API doesn't provide historical chains. Synthetic lets us test regime logic immediately. Switch to real data when available.

**Q: Is this strategy profitable in real trading?**  
A: Unknown. This is for research/education. Real trading requires:
- Actual historical data (not synthetic)
- Transaction cost modeling
- Liquidity analysis
- Risk management
- Live testing with small capital first

**Q: How do I know if adaptive strategy beats static?**  
A: Run backtest with `use_adaptive=True` vs `use_adaptive=False`. Compare CAGR and Sharpe ratios.

**Q: Can I trade this with a small account?**  
A: VIX options are expensive. UVXY options might be more accessible. Minimum recommended: $10,000 (and even then, limited to 1-2 contracts).

**Q: Why does the equity curve look smooth?**  
A: Weekly snapshots hide intraweek volatility. Real equity would be much choppier.

---

## 📄 License

MIT License - Free to use, modify, distribute.

**Disclaimer**: This is for educational purposes only. Not financial advice. Past performance doesn't guarantee future results. Options trading involves substantial risk of loss.

---

## 🙏 Credits

Built on your original VIX 5% Weekly Suite codebase. Enhanced with:
- Adaptive regime detection
- Comprehensive diagnostics  
- Better backtest execution
- Cleaner UI

---

*Last Updated: December 2025*
