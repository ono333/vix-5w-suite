#!/usr/bin/env python3
"""
Fix email to show ALL 5 variants for paper testing mode.
Active variants shown in green, suppressed in gray with reason.
"""

from pathlib import Path
import re

APP_FILE = Path("app.py")

# New email function that shows ALL variants
NEW_EMAIL_FUNCTION = '''
def send_signal_email_smtp(batch, recipient: str) -> tuple:
    """Send signal email via SMTP showing ALL variants for paper testing."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import os
    
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    
    if not smtp_user or not smtp_pass:
        return False, "SMTP credentials not configured. Set SMTP_USER and SMTP_PASS environment variables."
    
    regime = batch.regime_state
    
    # Count active vs suppressed
    active_count = sum(1 for v in batch.variants if regime.regime in v.active_in_regimes)
    suppressed_count = len(batch.variants) - active_count
    
    # Emoji based on regime
    regime_emoji = {
        "calm": "🟢", "declining": "🟡", "rising": "🟠", 
        "stressed": "🔴", "extreme": "⚫"
    }.get(regime.regime.value.lower(), "⚪")
    
    subject = f"{regime_emoji} VIX Signal: {regime.regime.value.upper()} ({regime.vix_percentile:.0%}) - {active_count} Active / {suppressed_count} Suppressed"
    
    # Calculate suggested contracts based on $250k account
    account_size = 250000
    
    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;font-size:14px;background:#fff;color:#333;padding:15px;max-width:800px;">
    
    <div style="text-align:center;border-bottom:2px solid #1f77b4;padding-bottom:10px;margin-bottom:15px;">
        <span style="font-size:22px;font-weight:bold;color:#1f77b4;">VIX 5% WEEKLY SUITE</span><br>
        <span style="font-size:14px;color:#666;">Paper Testing Signal • {batch.generated_at.strftime('%Y-%m-%d %H:%M UTC')}</span>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin-bottom:15px;">
        <tr>
            <td style="padding:10px;background:#f5f5f5;border:1px solid #ddd;width:25%;text-align:center;">
                <b>Regime</b><br><span style="font-size:18px;">{regime.regime.value.upper()}</span>
            </td>
            <td style="padding:10px;background:#f5f5f5;border:1px solid #ddd;width:25%;text-align:center;">
                <b>UVXY</b><br><span style="font-size:18px;">${regime.vix_level:.2f}</span>
            </td>
            <td style="padding:10px;background:#f5f5f5;border:1px solid #ddd;width:25%;text-align:center;">
                <b>Percentile</b><br><span style="font-size:18px;">{regime.vix_percentile:.0%}</span>
            </td>
            <td style="padding:10px;background:#f5f5f5;border:1px solid #ddd;width:25%;text-align:center;">
                <b>Confidence</b><br><span style="font-size:18px;">{regime.confidence:.0%}</span>
            </td>
        </tr>
    </table>
    
    <div style="font-size:16px;font-weight:bold;color:#1f77b4;margin-bottom:5px;">
        📊 ALL VARIANTS (Paper Testing Mode)
    </div>
    <div style="font-size:12px;color:#666;margin-bottom:15px;">
        {active_count} active for {regime.regime.value.upper()} regime | {suppressed_count} suppressed (shown for comparison)
    </div>
    """
    
    # Show ALL variants - active first, then suppressed
    for variant in sorted(batch.variants, key=lambda v: (regime.regime not in v.active_in_regimes, v.role.value)):
        is_active = regime.regime in variant.active_in_regimes
        
        # Calculate position sizing
        alloc_dollars = account_size * variant.alloc_pct
        est_risk_per_contract = variant.long_strike_offset * 100
        if est_risk_per_contract > 0:
            suggested_contracts = max(1, min(50, int(alloc_dollars / est_risk_per_contract)))
        else:
            suggested_contracts = 1
        total_risk = suggested_contracts * est_risk_per_contract
        
        if is_active:
            # Active variant - green styling
            header_bg = "#4CAF50"
            header_text = "#fff"
            status_icon = "✅"
            status_text = "ACTIVE"
            body_bg = "#f8fff8"
            border_color = "#4CAF50"
        else:
            # Suppressed variant - gray styling
            header_bg = "#9e9e9e"
            header_text = "#fff"
            status_icon = "⏸️"
            status_text = f"SUPPRESSED (not for {regime.regime.value.upper()})"
            body_bg = "#f5f5f5"
            border_color = "#ccc"
        
        # Get active regimes for this variant
        active_regimes = ", ".join([r.value.upper() for r in variant.active_in_regimes])
        
        html += f"""
        <div style="border:2px solid {border_color};margin-bottom:12px;border-radius:6px;overflow:hidden;">
            <div style="background:{header_bg};color:{header_text};padding:8px 12px;font-weight:bold;display:flex;justify-content:space-between;">
                <span>{status_icon} {get_variant_display_name(variant.role)}</span>
                <span style="font-size:12px;font-weight:normal;">{status_text}</span>
            </div>
            <div style="padding:12px;background:{body_bg};">
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                    <tr>
                        <td style="padding:4px;width:50%;"><b>Entry Threshold:</b> ≤{variant.entry_percentile:.0%} percentile</td>
                        <td style="padding:4px;width:50%;"><b>Long Strike:</b> UVXY +{variant.long_strike_offset}pts OTM</td>
                    </tr>
                    <tr>
                        <td style="padding:4px;"><b>Long DTE:</b> {variant.long_dte_weeks}w</td>
                        <td style="padding:4px;"><b>Short Strike:</b> UVXY +{variant.short_strike_offset}pts OTM</td>
                    </tr>
                    <tr>
                        <td style="padding:4px;"><b>Short DTE:</b> {variant.short_dte_weeks}w</td>
                        <td style="padding:4px;"><b>Roll Threshold:</b> {variant.roll_dte_days}d</td>
                    </tr>
                    <tr style="background:{'#e8f5e9' if is_active else '#eeeeee'};">
                        <td style="padding:6px;"><b>Allocation:</b> {variant.alloc_pct:.1%} (${alloc_dollars:,.0f})</td>
                        <td style="padding:6px;"><b>Suggested:</b> {suggested_contracts} contracts</td>
                    </tr>
                    <tr style="background:{'#e8f5e9' if is_active else '#eeeeee'};">
                        <td style="padding:6px;"><b>Target:</b> +{variant.tp_pct:.0%}</td>
                        <td style="padding:6px;"><b>Stop:</b> -{variant.sl_pct:.0%}</td>
                    </tr>
                    <tr>
                        <td colspan="2" style="padding:6px;color:#666;font-size:12px;">
                            <b>Est. Max Risk:</b> ${total_risk:,.0f} ({total_risk/account_size:.1%}) | 
                            <b>Active in:</b> {active_regimes}
                        </td>
                    </tr>
                </table>
            </div>
        </div>
        """
    
    html += f"""
    <div style="margin-top:20px;padding:12px;background:#e3f2fd;border-radius:4px;font-size:12px;">
        <b>📋 Paper Testing Notes:</b><br>
        • All 5 variants shown for observation and comparison<br>
        • Suppressed variants would not trade in live mode for this regime<br>
        • Track all variants to validate regime-based filtering logic<br>
        • Account basis: ${account_size:,}
    </div>
    
    <div style="margin-top:15px;padding-top:10px;border-top:1px solid #ddd;color:#888;font-size:11px;text-align:center;">
        VIX 5% Weekly Suite - Paper Testing Signal | {batch.generated_at.strftime('%Y-%m-%d %H:%M UTC')}
    </div>
    </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        return True, f"Email sent to {recipient}"
    except Exception as e:
        return False, f"Email failed: {str(e)}"
'''

def patch_app():
    content = APP_FILE.read_text()
    
    # Find and replace the send_signal_email_smtp function
    # Pattern to match the entire function
    pattern = r'def send_signal_email_smtp\(batch, recipient: str\) -> tuple:.*?(?=\ndef [a-z_]+\(|\nclass |\n# ===|\Z)'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Replace the function
        content = content[:match.start()] + NEW_EMAIL_FUNCTION.strip() + "\n\n" + content[match.end():]
        APP_FILE.write_text(content)
        print("✅ Email function updated to show ALL 5 variants")
        print("   - Active variants: green with ✅")
        print("   - Suppressed variants: gray with ⏸️")
        print("   - Shows which regimes each variant is active in")
        print("   - Includes allocation and contract sizing for all")
        return True
    else:
        print("❌ Could not find send_signal_email_smtp function")
        print("   Looking for function manually...")
        
        if "def send_signal_email_smtp" in content:
            print("   Function exists but pattern didn't match")
            print("   You may need to manually replace it")
        else:
            print("   Function not found - may need to add it")
        return False

if __name__ == "__main__":
    patch_app()
