@echo off
rem CR1b chain (QuantMind 10-14 CR1b approval; V4_COMBO_V1_CR1 s3; contract V4). V4 TRAIN only; DEV / OOS not read.
rem cr1b_calib (combo on A, residualised plant s_perp8, alpha p8/p32, visible + rank<=10 pre-run checks) ->
rem smoke p32 16 updates + gate -> PC p8 -> round pc (B half) -> NULL -> round null (bar, PC verdict) -> REAL ->
rem round real -> stop. Same budget %UPD% updates for PC / NULL / REAL. One GPU run at a time.
rem Refuses if any CR1b output exists. A failed pre-run check stops here and goes to the user.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\CR1B.log
set UPD=256
set BASE=--batch 128 --tiles 16 --workers 14 --max-hours 12
set OMP_NUM_THREADS=4
if exist "%A%\CR1B\CR1B_ALPHA.json" (
  echo === REFUSED %DATE% %TIME%: CR1B_ALPHA.json exists >> %LOG%
  exit /b 3
)
for %%R in (CR1B_SMOKE_P32 CR1B_PC_P8 CR1B_NULL CR1B_REAL) do (
  if exist "%A%\%%R\TRAIN_RECEIPT_V1.json" (
    echo === REFUSED %DATE% %TIME%: %%R receipt exists >> %LOG%
    exit /b 3
  )
  if exist "%A%\%%R\checkpoint_*.pt" (
    echo === REFUSED %DATE% %TIME%: %%R checkpoints exist >> %LOG%
    exit /b 3
  )
)
echo === CR1B START %DATE% %TIME% > %LOG%
echo === CALIB %DATE% %TIME% >> %LOG%
%PY% -I cr1b_calib.py >> %LOG% 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo === CALIB rc=%RC%: pre-run check failed ^(6 alpha, 7 not visible, 8 orth, 9 rank^>10^), NOT LAUNCHED, to user >> %LOG%
  exit /b 7
)
echo === SMOKE %DATE% %TIME% >> %LOG%
%PY% -I cr1b_trainer.py CR1B_SMOKE_P32 --mode p32 --updates 16 %BASE% >> %LOG% 2>&1
echo === SMOKE rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I cr1_smoke_gate.py CR1B_SMOKE_P32 --updates 16 >> %LOG% 2>&1
if errorlevel 1 (
  echo === SMOKE GATE FAILED, PC not started >> %LOG%
  exit /b 5
)
echo === PC %DATE% %TIME% >> %LOG%
%PY% -I cr1b_trainer.py CR1B_PC_P8 --mode p8 --updates %UPD% %BASE% >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\CR1B_PC_P8\TRAIN_RECEIPT_V1.json" (
  echo === PC has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I cr1b_round.py CR1B_PC_P8 --mode pc >> %LOG% 2>&1
if errorlevel 1 (
  echo === ROUND PC rc=%ERRORLEVEL%: no frozen formula with corr ^> 0.7, PC NOT FOUND, chain stops >> %LOG%
  exit /b 20
)
echo === NULL %DATE% %TIME% >> %LOG%
%PY% -I cr1b_trainer.py CR1B_NULL --mode null --updates %UPD% %BASE% >> %LOG% 2>&1
echo === NULL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\CR1B_NULL\TRAIN_RECEIPT_V1.json" (
  echo === NULL has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I cr1b_round.py CR1B_NULL --mode null --pc CR1B_PC_P8 >> %LOG% 2>&1
if errorlevel 1 (
  echo === ROUND NULL rc=%ERRORLEVEL%: PC not above the NULL bar, PC NOT FOUND, chain stops >> %LOG%
  exit /b 21
)
echo === REAL %DATE% %TIME% >> %LOG%
%PY% -I cr1b_trainer.py CR1B_REAL --mode real --updates %UPD% %BASE% >> %LOG% 2>&1
echo === REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\CR1B_REAL\TRAIN_RECEIPT_V1.json" (
  echo === REAL has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I cr1b_round.py CR1B_REAL --mode real --null CR1B_NULL >> %LOG% 2>&1
echo === ROUND REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
echo === CR1B END %DATE% %TIME% >> %LOG%
