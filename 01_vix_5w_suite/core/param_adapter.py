def adapt_to_v2(params, vix_weekly):
    return dict(
        vix_weekly=vix_weekly,
        mode="diagonal",
        alloc_pct=params["alloc_frac"],
        entry_lookback_weeks=52,
        entry_percentile=params["entry_pct"],
        r=params["risk_free"],
        realistic_mode=params["realism"],
        realism_level=params["realism_level"],
        otm_pts=params["otm_pts"],
        target_mult=params["target_mult"],
        sigma_mult=params["sigma_mult"],
        long_dte_weeks=params["long_dte_weeks"],
        fee_per_contract=params["fee_per_contract"],
        exit_mult=params["exit_mult"]
    )

def adapt_to_v3(params, vix_weekly):
    vix_pct = compute_vix_percentile(vix_weekly, 52)
    return dict(
        vix_pct_series=vix_pct,
        entry_percentile=params["entry_pct"],
        otm_pts=params["otm_pts"],
        target_mult=params["target_mult"],
        sigma_mult=params["sigma_mult"],
        long_dte_weeks=params["long_dte_weeks"],
        fee_per_contract=params["fee_per_contract"],
        exit_mult=params["exit_mult"],
        realism=params["realism"],
        initial_capital=params["initial_capital"],
        alloc_pct=params["alloc_frac"],
    )