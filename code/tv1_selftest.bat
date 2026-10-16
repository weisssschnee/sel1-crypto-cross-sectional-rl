@echo off
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set OUT=D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\TV1_20261010
if exist %OUT%\TV1_SELFTEST.json (echo TV1_SELFTEST exists, refusing & exit /b 2)
%PY% -I tv1_selftest.py > %OUT%\selftest.log 2>&1
echo EXIT=%ERRORLEVEL% >> %OUT%\selftest.log
