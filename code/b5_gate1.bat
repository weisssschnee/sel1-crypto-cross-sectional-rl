@echo off
rem BASIN5 step 1 ruler check (G0 -> G1 -> G2, CPU, TRAIN 50 fixed tiles) -> B5_GATE1\DECISION.json.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_GATE1.log
echo === GATE1 %DATE% %TIME% > %LOG%
%PY% -I b5_gate1.py --workers 16 >> %LOG% 2>&1
echo === GATE1 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
