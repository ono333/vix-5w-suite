# systemd units (reference copies)

Live location: /etc/systemd/system/
These are version-controlled copies; editing here does nothing until installed.

## vix_shadow_marks.{service,timer}
Runs shadow_strategist.py --marks-only --force on Tue/Wed/Thu 15:45 ET.
Uses EnvironmentFile=/etc/vix_orchestrator.env for the Tradier token
(crontab could not read that root-owned file — hence the systemd timer).

Reinstall after a rebuild:
  sudo cp deploy/vix_shadow_marks.service /etc/systemd/system/
  sudo cp deploy/vix_shadow_marks.timer   /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now vix_shadow_marks.timer
  systemctl list-timers vix_shadow_marks.timer --no-pager
