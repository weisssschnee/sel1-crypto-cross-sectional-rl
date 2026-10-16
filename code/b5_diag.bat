@echo off
rem BASIN5 parallel diagnostics (regime 2x2, sector lead-lag), TRAIN only, CPU, not gating.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set OMP_NUM_THREADS=4
set B5_DIAG_V2=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_DIAG_V2.log
echo === DIAG %DATE% %TIME% > %LOG%
%PY% -I b5_diag_regime_sector.py >> %LOG% 2>&1
echo === DIAG rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
