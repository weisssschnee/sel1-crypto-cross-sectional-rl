@echo off
rem XS_SEL1 V2 resume: NULL --resume auto (module sha check) -> round null -> REAL -> round real (sel1_chain_v2 tails).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_V2_RESUME.log
echo === SEL1 V2 RESUME %DATE% %TIME% >> %LOG%
set /p FNULL=<"%O%\_flags_NULL_V2.txt"
set /p FREAL=<"%O%\_flags_REAL_V2.txt"
%PY% -I sel1_trainer.py SEL1_NULL %FNULL% --resume auto >> %LOG% 2>&1
echo === NULL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%O%\SEL1_NULL\TRAIN_RECEIPT_V1.json" (
  echo === NULL has no receipt, stops >> %LOG%
  exit /b 20
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
  exit /b 22
)
%PY% -I sel1_round.py SEL1_REAL --mode real --null SEL1_NULL >> %LOG% 2>&1
echo === ROUND REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
echo === SEL1 V2 RESUME END %DATE% %TIME% >> %LOG%
