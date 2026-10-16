@echo off
rem ENR1 enrichment table (order ENR1 step 2, ADR-0012, PREREG sha16 1db50993ce274590). CPU only, V4 TRAIN only.
rem Refuses to start if the table exists; enr1.py refuses too.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set ENR1_TEST=
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\ENR1.log
if exist D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\ENR1\ENR1_TABLE.json (
  echo === ENR1 REFUSED: table exists %DATE% %TIME% >> %LOG%
  exit /b 3
)
echo === ENR1 %DATE% %TIME% > %LOG%
%PY% -I enr1.py >> %LOG% 2>&1
echo === ENR1 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
