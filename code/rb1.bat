@echo off
rem RB1 rule book on V4 TRAIN (order ENR1 step 3). CPU only; no FROZEN, no DEV. rb1.py refuses if RB1_TRAIN.json exists.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set ENR1_TEST=
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\RB1.log
if exist D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\ENR1\RB1_TRAIN.json (
  echo === RB1 REFUSED: receipt exists %DATE% %TIME% >> %LOG%
  exit /b 3
)
echo === RB1 %DATE% %TIME% > %LOG%
%PY% -I rb1.py >> %LOG% 2>&1
echo === RB1 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
