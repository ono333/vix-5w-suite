# Troubleshooting Guide

## Problem: Equity Still Flat (No Trades)

### Quick Diagnostic Checklist

1. **Check the diagnostic panel in the app** (appears automatically when trades = 0)

2. **Look for these specific errors**:

   #### "No entry signals detected"
   ```
   Cause: VIX percentile never low enough to trigger entry
   Solution: Lower entry thresholds in regime configs, or extend date range
   ```

   #### "Entry signals failed: insufficient_capital"
   ```
   Cause: Position size too small (< 1 contract) or not enough cash
   Solution: 
   - Increase initial_capital
   - Reduce otm_pts (cheaper strikes)
   - Increase alloc_pct
   ```

   #### "Entry signals failed: invalid_long_price"
   ```
   Cause: Black-Scholes pricing returning NaN/inf
   Solution: Check sigma_mult parameter (should be 0.5-2.0)
   ```

   #### "VIX percentile calculation failed (all NaN)"
   ```
   Cause: Not enough historical data
   Solution: Extend start_date to at least 52 weeks before first trade
   ```

### Step-by-Step Debugging

#### Step 1: Verify Data Loading
```python
# In app.py, add after loading data:
st.write("VIX data sample:")
st.write(vix_weekly.head())
st.write(f"Total weeks: {len(vix_weekly)}")
st.write(f"Date range: {vix_weekly.index[0]} to {vix_weekly.index[-1]}")
```

Expected: Should show 50+ weeks of VIX data

#### Step 2: Check Percentile Calculation
```python
# In app.py, add after creating regime_adapter:
st.write("Percentile sample:")
st.write(regime_adapter.vix_percentile.tail(20))
st.write(f"Valid percentiles: {regime_adapter.vix_percentile.notna().sum()}")
```

Expected: Should have valid percentiles (not all NaN)

#### Step 3: Inspect Entry Signals
```python
# In app.py, after backtest:
entry_signals = bt_results.get("entry_signals", [])
st.write(f"Entry signals detected: {len(entry_signals)}")
if entry_signals:
    st.write("First 5 entry signals:")
    st.write(pd.DataFrame(entry_signals).head())
```

Expected: Should have at least a few entry signals

#### Step 4: Check Failed Trades
```python
# In app.py, after backtest:
no_trade_reasons = bt_results.get("no_trade_reasons", [])
st.write(f"Failed trade attempts: {len(no_trade_reasons)}")
if no_trade_reasons:
    # Group by reason
    reasons = pd.DataFrame(no_trade_reasons)
    st.write("Failure reasons:")
    st.write(reasons["reason"].value_counts())
```

Expected: Shows why entry signals didn't become trades

---

## Problem: Trades Executing But Poor Performance

### Diagnostic Questions

#### 1. Too Many Losses?
**Check**: Win rate in performance metrics  
**Normal Range**: 40-60% for VIX strategies  
**If < 30%**: Strategy entering at wrong times
- Solution: Adjust entry_percentile thresholds
- Consider tighter stop losses (lower exit_mult)

#### 2. Small Profit, Large Losses?
**Check**: Trade log - compare winning vs losing trade sizes  
**Problem**: Asymmetric risk/reward  
**Solution**: 
- Tighter stops (exit_mult closer to 1.0)
- Higher profit targets (target_mult > 1.5)

#### 3. Too Few Trades?
**Check**: Trades count in performance metrics  
**Normal Range**: 10-50 trades per year  
**If < 5 per year**: Entry conditions too strict
- Solution: Raise entry_percentile thresholds
- Consider multiple regime ranges

#### 4. Too Many Trades?
**Check**: Trades count, average duration  
**Normal Range**: Hold time > 2 weeks  
**If < 1 week**: Exiting too quickly
- Solution: Wider stops, higher profit targets

---

## Problem: App Crashing or Errors

### Import Errors

```
ModuleNotFoundError: No module named 'regime_adapter'
```

**Solution**: 
```bash
# Make sure all files are in the correct directories
ls -la core/
# Should see:
# regime_adapter.py
# adaptive_backtester.py
# backtester.py
# data_loader.py
# etc.
```

### SciPy Not Found

```
ModuleNotFoundError: No module named 'scipy'
```

**Solution**:
```bash
pip install scipy
```

### YFinance Download Fails

```
Error: VIX data download failed
```

**Solution**:
1. Check internet connection
2. Try different date range (Yahoo sometimes has gaps)
3. Use backup VIX data if available

---

## Problem: Results Don't Match Expectations

### Unrealistic High Returns

**Symptoms**: CAGR > 100%, Win rate = 100%  
**Cause**: Synthetic pricing too optimistic  
**Solutions**:
1. Lower `realism` parameter (try 0.7-0.8)
2. Increase slippage assumptions
3. Add position size limits
4. Wait for real historical data

### Excessive Drawdowns

**Symptoms**: Max DD > 50%  
**Possible Causes**:
1. Overleveraged (alloc_pct too high)
2. No stop losses working (check exit_mult)
3. Holding through expiration (check DTE management)

**Solutions**:
- Reduce alloc_pct (0.005-0.01 is safer)
- Tighter stops (exit_mult = 0.5-0.6)
- Don't let options expire worthless

---

## Problem: Regime Detection Not Working

### All Weeks in Same Regime

**Check**: Regime history in diagnostics panel  
**Expected**: Should see transitions between regimes  
**If stuck in one regime**:
- VIX range too narrow in selected period
- Extend date range to include 2008, 2020 (high VIX)
- Adjust percentile_range boundaries in configs

### Too Many Regime Switches

**Check**: Regime duration in history  
**Expected**: Regimes should last weeks/months  
**If switching every week**:
- lookback_weeks too short (increase to 52+)
- VIX too noisy (add smoothing)

---

## Problem: Massive API Not Working

**This is expected!** The app uses **synthetic pricing** because:

1. Massive doesn't provide historical chains via REST
2. The endpoint `/v1/options/historical-chains` doesn't exist
3. Even if it did, it would require custom data arrangement

**Solution**: Use synthetic pricing mode. When real historical data becomes available:

```python
# In backtester.py, replace this:
lp = bs_call_price(S, strike_long, r, sigma_eff, long_dte_weeks / 52.0)

# With this:
lp = get_real_option_price(symbol, trade_date, strike_long, expiry_date)
```

---

## Advanced Debugging

### Enable Verbose Mode

In the sidebar: Check "Verbose diagnostics"

This prints to console:
```
Week 45: ENTRY - VIX=18.5, pct=15%, mode=diagonal, qty=5, strike=28.5
Week 52: EXIT - profit_target_1.5x, PnL: 2,450.00
```

### Inspect Regime Transitions

```python
# In app.py, add:
regime_df = pd.DataFrame(regime_adapter.regime_history)
st.write("Regime transitions:")
st.write(regime_df[regime_df["regime"] != regime_df["regime"].shift()])
```

Shows exactly when/why regime changed

### Validate Option Pricing

```python
# In backtester.py, add logging:
if verbose:
    print(f"BS Call Price: S={S}, K={strike_long}, r={r}, "
          f"sigma={sigma_eff}, T={long_dte_weeks/52}, price={lp}")
```

Helps catch NaN/inf pricing issues

---

## Getting Help

### Before Asking for Help

1. ✅ Run diagnostics panel (built into app)
2. ✅ Check this troubleshooting guide
3. ✅ Enable verbose mode and review console output
4. ✅ Export regime history and trade log CSVs

### What to Include in Bug Report

1. **Diagnostic output** (from app panel)
2. **Date range used** (start_date, end_date)
3. **Parameter values** (initial_capital, regime configs, etc.)
4. **Screenshots** of results
5. **Console output** (if verbose mode enabled)

### Common Non-Issues

These are **not bugs**:

❌ "Equity curve doesn't match my live trading"  
→ This uses synthetic pricing, not real fills

❌ "Returns too high to be realistic"  
→ Lower realism parameter, add slippage

❌ "Strategy loses money in 2020"  
→ That's valid! Not all strategies work in all periods

❌ "Massive API doesn't have historical chains"  
→ Correct, that's why we use synthetic pricing

---

## Performance Optimization

### App Running Slow?

```python
# Cache data loading
@st.cache_data
def load_vix_weekly(start, end):
    # ... existing code ...

# Cache regime adapter
@st.cache_resource
def create_regime_adapter(vix_weekly, lookback):
    return RegimeAdapter(vix_weekly, lookback)
```

### Backtest Taking Too Long?

- Reduce date range (test on 1-2 years first)
- Simplify regime configs (fewer regimes)
- Remove verbose logging

---

*Last Updated: January 2026*
