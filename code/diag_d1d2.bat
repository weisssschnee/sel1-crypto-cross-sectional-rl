@echo off
rem D1/D2 regime diagnostics + pre-registered H1/H2 on V4 2021-23 (QuantMind 10-14), TRAIN only, CPU, not gating.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set LOG=D:\ChengboRemote\runtime\basin3_h1\DIAG_D1D2.log
if exist D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\DIAG_D1D2\PREREG_H1H2.json (
  echo === D1D2 REFUSED: prereg exists %DATE% %TIME% >> %LOG%
  exit /b 3
)
echo === D1D2 %DATE% %TIME% > %LOG%
%PY% -I diag_d1d2.py >> %LOG% 2>&1
echo === D1D2 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
