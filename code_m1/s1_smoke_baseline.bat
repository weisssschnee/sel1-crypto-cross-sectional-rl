@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
%PY% dev_score.py --run S1_SMOKE_NOTRAIN --months 2026-03 --baseline-only --depth-dir D:\ChengboRemote\runtime\p1 --out D:\ChengboRemote\runtime\m1_stage1\smoke_baseline_2026-03.json
