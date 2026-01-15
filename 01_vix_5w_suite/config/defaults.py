import datetime as dt

DEFAULTS = {
    "initial_capital": 250_000.0,
    "alloc_pct": 0.01,
    "long_dte_weeks": 26,
    "risk_free": 0.03,
    "fee_per_contract": 0.65,
    "entry_percentile": 0.10,
    "otm_points": 10.0,
    "target_multiple": 1.2,
    "sigma_mult": 0.5,
    "start_date": dt.date(2015, 1, 1),
    "end_date": dt.date.today(),
    "entry_mode": "percentile",   # or 'osc_roc'
    "position_structure": "diagonal",  # or 'long_only'
}
