@echo off
rem BASIN5 step 1 pipeline test: b5_gate1.py with small pools (--limit 4), 4 workers. Builds the
rem fixed-tile cache and the p16 calibration (both reused by the real gate). Output is overwritten
rem by b5_gate1.bat; nothing in it is used for a decision.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_GATE1_TEST.log
echo === GATE1 TEST %DATE% %TIME% > %LOG%
%PY% -I b5_gate1.py --limit 4 --workers 4 >> %LOG% 2>&1
echo === GATE1 TEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
