@echo off
rem XS_SEL1 chain (order XS_SEL1 s4, supplement 10-15 late night). V4 TRAIN only; DEV / OOS not read.
rem launch manifest (from SEL1_NULLPLANT chosen cell + SEL1_REWARD_RULE) -> smoke 8 updates (PC flags) + gate + round
rem script test on the smoke hall -> PC -> round pc -> NULL -> round null (T_null, PC verdict) -> REAL -> round real.
rem One GPU run at a time. Refuses if any output exists.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_CHAIN.log
set OMP_NUM_THREADS=4
if exist "%O%\SEL1_LAUNCH_MANIFEST.json" (
  echo === REFUSED %DATE% %TIME%: SEL1_LAUNCH_MANIFEST.json exists >> %LOG%
  exit /b 3
)
for %%R in (SEL1_SMOKE SEL1_PC SEL1_NULL SEL1_REAL) do (
  if exist "%O%\%%R" (
    echo === REFUSED %DATE% %TIME%: %%R exists >> %LOG%
    exit /b 3
  )
)
echo === SEL1 CHAIN START %DATE% %TIME% > %LOG%
%PY% -I sel1_launch_manifest.py >> %LOG% 2>&1
if errorlevel 1 (
  echo === MANIFEST rc=%ERRORLEVEL%, chain stops >> %LOG%
  exit /b 7
)
set /p FPC=<"%O%\_flags_PC.txt"
set /p FNULL=<"%O%\_flags_NULL.txt"
set /p FREAL=<"%O%\_flags_REAL.txt"
set /p FSMOKE=<"%O%\_flags_SMOKE.txt"
echo === SMOKE %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer.py SEL1_SMOKE %FSMOKE% >> %LOG% 2>&1
echo === SMOKE rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_smoke_gate.py SEL1_SMOKE --updates 8 >> %LOG% 2>&1
if errorlevel 1 (
  echo === SMOKE GATE FAILED, PC not started >> %LOG%
  exit /b 5
)
for %%M in ("pc" "null --pc SEL1_SMOKE" "real --null SEL1_SMOKE") do (
  %PY% -I sel1_round.py SEL1_SMOKE --mode %%~M --top 10 >> %LOG% 2>&1
  if errorlevel 1 if not errorlevel 20 (
    echo === ROUND TEST %%~M crashed, PC not started >> %LOG%
    exit /b 5
  )
)
echo === PC %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer.py SEL1_PC %FPC% >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%O%\SEL1_PC\TRAIN_RECEIPT_V1.json" (
  echo === PC has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I sel1_round.py SEL1_PC --mode pc >> %LOG% 2>&1
if errorlevel 1 (
  echo === ROUND PC rc=%ERRORLEVEL%: no hall entry with corr ^> 0.7, PC NOT FOUND, chain stops >> %LOG%
  exit /b 20
)
echo === NULL %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer.py SEL1_NULL %FNULL% >> %LOG% 2>&1
echo === NULL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%O%\SEL1_NULL\TRAIN_RECEIPT_V1.json" (
  echo === NULL has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I sel1_round.py SEL1_NULL --mode null --pc SEL1_PC >> %LOG% 2>&1
if errorlevel 1 (
  echo === ROUND NULL rc=%ERRORLEVEL%: PC not above T_null, PC NOT FOUND, chain stops >> %LOG%
  exit /b 21
)
echo === REAL %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer.py SEL1_REAL %FREAL% >> %LOG% 2>&1
echo === REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%O%\SEL1_REAL\TRAIN_RECEIPT_V1.json" (
  echo === REAL has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I sel1_round.py SEL1_REAL --mode real --null SEL1_NULL >> %LOG% 2>&1
echo === ROUND REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
echo === SEL1 CHAIN END %DATE% %TIME% >> %LOG%
