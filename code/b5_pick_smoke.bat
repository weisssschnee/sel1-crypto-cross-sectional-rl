@echo off
rem BASIN5 reward pick, CPU smoke (4 formulas per pool) into BASIN5_20261012\smoke.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
%PY% -I b5_reward_pick.py --smoke 4 > D:\ChengboRemote\runtime\basin3_h1\B5_PICK_SMOKE.log 2>&1
echo rc=%errorlevel% >> D:\ChengboRemote\runtime\basin3_h1\B5_PICK_SMOKE.log
