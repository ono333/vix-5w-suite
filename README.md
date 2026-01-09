# Position-Aware VIX 5% Weekly Suite

This update transforms the system from a **signal broadcaster** into a **regime-aware, position-managed trading system**.

## What's Included

| File | Purpose |
|------|---------|
| `trade_log.py` | Position storage & tracking (backend) |
| `daily_signal.py` | Position-aware email generator |
| `position_ui.py` | Streamlit UI components |
| `add_position_page.py` | Script to add Position Manager page |
| `add_test_positions.py` | Add sample positions for testing |
| `clear_positions.py` | Clear all positions |

## What Changed

### Before (Signal Broadcast)
- Email showed "Entry ≤25% percentile" for all variants
- No awareness of open positions
- Could suggest entries when positions already exist
- Entry percentiles are pre-trade information, useless once positions exist

### After (Position-Aware)
- Email reads from trade log to detect open positions
- **MANAGEMENT MODE** for variants with positions:
  - Shows P&L, DTE remaining
  - Shows target/stop prices
  - Suggests: hold, take profit, stop loss, roll, close
- **ENTRY MODE** for variants without positions:
  - Shows suggested entry credit
  - Shows target/stop prices
  - Shows allocation and contract sizing
- **REGIME DRIFT WARNING** when positions opened in different regime

## Installation

```bash
cd ~/Downloads
unzip -o position_aware_system.zip

cd ~/PRR/01_vix_5w_suite

# Backup existing files
cp trade_log.py trade_log.py.backup 2>/dev/null
cp daily_signal.py daily_signal.py.backup 2>/dev/null

# Install all files
cp ~/Downloads/position_aware_system/*.py .

# Add Position Manager page to app.py
python3 add_position_page.py

# Clear cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Restart Streamlit
streamlit run app.py
```

## New UI Components

### 1. Position Manager Page (Streamlit)

A new page in the sidebar with three tabs:

**🔄 Open Positions Tab**
- Shows all open positions with:
  - Entry credit & contracts
  - Current P&L ($ and %)
  - DTE remaining
  - Target & stop prices
  - Action suggestions (Hold, Take Profit, Stop Loss, Roll)
- Close position button with exit price input
- Update prices form to recalculate P&L

**📝 Record Trade Tab**
- Form to record actual fills:
  - Variant selection (filters out those with positions)
  - Entry credit received
  - Contracts
  - Strike price
  - Expiration date
  - Target/stop percentages
- Auto-calculates target & stop prices
- Saves to persistent trade log

**📈 Performance Tab**
- Overall metrics: Total trades, win rate, realized P&L
- Per-variant breakdown: Trades, win rate, P&L
- Complete trade history table

### 2. Signal Suppression

Entry signals are automatically suppressed for variants with open positions:

```python
from position_ui import should_suppress_entry

if should_suppress_entry("V1_INCOME_HARVESTER"):
    # Don't show entry signal - position exists
```

### 3. Position-Aware Email

Email now has three sections:
- **🔄 OPEN POSITIONS** - Management mode with P&L, DTE, actions
- **🎯 ENTRY CANDIDATES** - Entry mode with targets
- **🔬 PAPER TEST** - Variants to observe only

## Testing

### 1. Add test positions (to see management mode)
```bash
python3 add_test_positions.py
```

This adds:
- V1 Income Harvester (in profit)
- V5 Regime Allocator (slight loss)

### 2. Run signal generator
```bash
python3 daily_signal.py --dry-run
```

Expected output:
```
🔄 OPEN POSITIONS (2):
   • V1 Income Harvester: $+300 P&L, 23 DTE
     → ✋ HOLD - On track
   • V5 Regime Allocator: $-100 P&L, 16 DTE
     → ✋ HOLD - On track

🎯 ENTRY CANDIDATES (0):
   (None - V1 and V5 have positions)

🔬 PAPER TEST (3):
   🔬 V2 Mean Reversion (Activates in: DECLINING)
   🔬 V3 Shock Absorber (Activates in: RISING, STRESSED, EXTREME)
   🔬 V4 Tail Hunter (Activates in: EXTREME)
```

### 3. Send real email
```bash
export SMTP_USER="your@gmail.com"
export SMTP_PASS="your_app_password"
python3 daily_signal.py
```

### 4. Clear test positions (to see entry mode)
```bash
python3 clear_positions.py
```

Then run `--dry-run` again to see entry candidates.

## Email Sections

### 🔄 OPEN POSITIONS (Management Mode)
For each variant with an open position:
- Current P&L (dollars and %)
- DTE remaining
- Entry price
- Target exit price
- Stop loss price
- Action suggestion (hold, take profit, stop loss, roll, close)

### 🎯 ENTRY CANDIDATES (Entry Mode)
For recommended variants without positions:
- Entry trigger (percentile)
- Strike offset and DTE
- Suggested entry credit
- Target exit price
- Stop loss price
- Allocation and contract sizing

### 🔬 PAPER TEST ONLY
For variants not recommended in current regime:
- Basic parameters
- Which regimes they activate in
- Warning that they're for observation only

### ⚠️ REGIME DRIFT WARNING
When positions exist that were opened in a different regime:
- Lists affected positions
- Original regime vs current regime
- Suggests considering closing

## Trade Log Storage

Positions are stored in: `~/.vix_suite/trade_log.json`

The trade log tracks:
- Open positions by variant
- Entry/exit prices and dates
- P&L calculations
- Target/stop prices
- Regime at entry vs current

## Key Design Principles

1. **Email reflects STATE, not signals**
   - Shows what you HAVE, not just what you COULD do
   
2. **One position per variant**
   - Cannot enter same variant twice
   - Must close before re-entering
   
3. **Target/stop computed at entry**
   - Short premium: target = entry × (1 - target_pct)
   - Short premium: stop = entry × (1 + stop_pct)
   
4. **Regime awareness**
   - Tracks regime at entry
   - Warns when regime changes

## Files

- `trade_log.py` - Position tracking and persistence
- `daily_signal.py` - Position-aware signal generator
- `add_test_positions.py` - Add sample positions for testing
- `clear_positions.py` - Clear all positions
