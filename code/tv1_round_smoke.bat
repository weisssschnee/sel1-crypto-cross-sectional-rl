@echo off
rem tv1_round.py smoke on the two self-test smoke dirs (16 per family); output next to them.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\tv1_round_smoke.log
echo === START %DATE% %TIME% > %LOG%
%PY% -I tv1_round.py TV1_SMOKE_DENSE_123516 --trainer dense --mode p8 --per-family 16 >> %LOG% 2>&1
echo === dense rc=%ERRORLEVEL% >> %LOG%
%PY% -I tv1_round.py TV1_SMOKE_OLD_123516 --trainer old --mode p32 --per-family 16 >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
