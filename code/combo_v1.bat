@echo off
rem COMBO_V1 (order V4_COMBO_V1_CR1 s2.2, contract V4): stage 0 layer/oracle checks, stage 1 walk-forward
rem signs 2022-25 gate, then freeze and DEV once. combo_v1.py refuses to start if any output exists. CPU only.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set COMBO_TEST=
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=8
set LOG=D:\ChengboRemote\runtime\basin3_h1\COMBO_V1.log
if exist D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\COMBO_V1\COMBO_V1_TRAIN.json (
  echo === COMBO1 REFUSED: receipt exists %DATE% %TIME% >> %LOG%
  exit /b 3
)
echo === COMBO1 %DATE% %TIME% > %LOG%
%PY% -I combo_v1.py >> %LOG% 2>&1
echo === COMBO1 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
