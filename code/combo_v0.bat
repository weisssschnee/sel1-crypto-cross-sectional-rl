@echo off
rem COMBO_V0 (ADR-0010): stage 0 sign check, stage 1 2024->2025 holdout, then freeze and DEV once.
rem combo_v0.py refuses to start if FROZEN_COMBO_V0.json or COMBO_V0_DEV.json exists. CPU only.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set COMBO_TEST=
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=8
set LOG=D:\ChengboRemote\runtime\basin3_h1\COMBO_V0.log
if exist D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\COMBO_V0\FROZEN_COMBO_V0.json (
  echo === COMBO REFUSED: frozen file exists %DATE% %TIME% >> %LOG%
  exit /b 3
)
echo === COMBO %DATE% %TIME% > %LOG%
%PY% -I combo_v0.py >> %LOG% 2>&1
echo === COMBO rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
