@echo off
rem XS_SEL1 chain v2 (supplement 10-15 morning: --combo-sweep). V4 TRAIN only; DEV / OOS not read.
rem manifest v2 smoke flags -> smoke 8 updates + gate + round script test -> manifest v2 final (budget from the
rem smoke seconds) -> PC -> round pc -> NULL -> round null -> REAL -> round real. One GPU run at a time.
rem Refuses if the v2 manifest or any v2 run directory exists. Old PC kept as SEL1_PC_V1_COMBOTOKEN.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_CHAIN_V2.log
set OMP_NUM_THREADS=4
if exist "%O%\SEL1_LAUNCH_MANIFEST_V2.json" (
  echo === REFUSED %DATE% %TIME%: SEL1_LAUNCH_MANIFEST_V2.json exists >> %LOG%
  exit /b 3
)
for %%R in (SEL1_SMOKE_V2 SEL1_PC SEL1_NULL SEL1_REAL) do (
  if exist "%O%\%%R" (
    echo === REFUSED %DATE% %TIME%: %%R exists >> %LOG%
    exit /b 3
  )
)
echo === SEL1 CHAIN V2 START %DATE% %TIME% > %LOG%
%PY% -I sel1_launch_manifest_v2.py --phase smoke >> %LOG% 2>&1
if errorlevel 1 (
  echo === MANIFEST SMOKE rc=%ERRORLEVEL%, chain stops >> %LOG%
  exit /b 7
)
set /p FSMOKE=<"%O%\_flags_SMOKE_V2.txt"
echo === SMOKE %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer.py SEL1_SMOKE_V2 %FSMOKE% >> %LOG% 2>&1
echo === SMOKE rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_smoke_gate.py SEL1_SMOKE_V2 --updates 8 >> %LOG% 2>&1
if errorlevel 1 (
  echo === SMOKE GATE FAILED, PC not started >> %LOG%
  exit /b 5
)
for %%M in ("pc" "null --pc SEL1_SMOKE_V2" "real --null SEL1_SMOKE_V2") do (
  %PY% -I sel1_round.py SEL1_SMOKE_V2 --mode %%~M --top 10 >> %LOG% 2>&1
  if errorlevel 1 if not errorlevel 20 (
    echo === ROUND TEST %%~M crashed, PC not started >> %LOG%
    exit /b 5
  )
)
%PY% -I sel1_launch_manifest_v2.py --phase final >> %LOG% 2>&1
if errorlevel 1 (
  echo === MANIFEST FINAL rc=%ERRORLEVEL%, chain stops >> %LOG%
  exit /b 7
)
set /p FPC=<"%O%\_flags_PC_V2.txt"
set /p FNULL=<"%O%\_flags_NULL_V2.txt"
set /p FREAL=<"%O%\_flags_REAL_V2.txt"
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
echo === SEL1 CHAIN V2 END %DATE% %TIME% >> %LOG%
