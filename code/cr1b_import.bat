@echo off
rem CR1b import + resid unit check only (no panel read).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
%PY% -I cr1b_unit.py > D:\ChengboRemote\runtime\basin3_h1\CR1B_UNIT.log 2>&1
