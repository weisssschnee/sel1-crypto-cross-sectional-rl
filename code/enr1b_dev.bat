@echo off
rem XS_SEL1 step 0: ENR1b LS cells, one DEV read (frozen FROZEN_ENR1b_LS.json). CPU only. enr1b_dev.py refuses a second run.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set CUDA_VISIBLE_DEVICES=-1
set LOG=D:\ChengboRemote\runtime\basin3_h1\ENR1B_LS_DEV.log
if exist D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\XS_SEL1\ENR1b_LS_DEV.json (
  echo === LSDEV REFUSED: output exists %DATE% %TIME% >> %LOG%
  exit /b 3
)
echo === LSDEV %DATE% %TIME% > %LOG%
%PY% -I enr1b_dev.py >> %LOG% 2>&1
echo === LSDEV rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
