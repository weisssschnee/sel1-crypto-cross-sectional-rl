@echo off
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set OUT=D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\TV1_20261010
if exist %OUT%\A_TABLE.json (echo A_TABLE exists, refusing & exit /b 2)
if not exist %OUT% mkdir %OUT%
%PY% -I m2h_snr_table.py > %OUT%\a_table.log 2>&1
echo EXIT=%ERRORLEVEL% >> %OUT%\a_table.log
