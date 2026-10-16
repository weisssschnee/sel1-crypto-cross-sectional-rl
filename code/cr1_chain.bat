@echo off
rem CR1 chain (order V4_COMBO_V1_CR1 s3, contract V4, ADR-0010 decision 3). V4 TRAIN only; DEV / OOS not read.
rem cr1_calib (combo on A, orthogonal plant, alpha p8/p32, ruler check) -> smoke p32 16 updates + gate ->
rem PC p8 -> round pc (B half) -> NULL -> round null (bar, PC verdict) -> REAL -> round real.
rem Same budget %UPD% updates for PC / NULL / REAL. One GPU run at a time. Refuses if any output exists.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\CR1.log
set UPD=256
set BASE=--batch 128 --tiles 16 --workers 14 --max-hours 12
set OMP_NUM_THREADS=4
if exist "%A%\CR1\CR1_ALPHA.json" (
  echo === REFUSED %DATE% %TIME%: CR1_ALPHA.json exists >> %LOG%
  exit /b 3
)
for %%R in (CR1_SMOKE_P32 CR1_PC_P8 CR1_NULL CR1_REAL) do (
  if exist "%A%\%%R\TRAIN_RECEIPT_V1.json" (
    echo === REFUSED %DATE% %TIME%: %%R receipt exists >> %LOG%
    exit /b 3
  )
  if exist "%A%\%%R\checkpoint_*.pt" (
    echo === REFUSED %DATE% %TIME%: %%R checkpoints exist >> %LOG%
    exit /b 3
  )
)
echo === CR1 START %DATE% %TIME% > %LOG%
echo === CALIB %DATE% %TIME% >> %LOG%
%PY% -I cr1_calib.py >> %LOG% 2>&1
if errorlevel 1 (
  echo === CALIB rc=%ERRORLEVEL%: ruler / orthogonality / alpha failed, chain stops ^(veto rule, to user^) >> %LOG%
  exit /b 7
)
echo === SMOKE %DATE% %TIME% >> %LOG%
%PY% -I cr1_trainer.py CR1_SMOKE_P32 --mode p32 --updates 16 %BASE% >> %LOG% 2>&1
echo === SMOKE rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I cr1_smoke_gate.py CR1_SMOKE_P32 --updates 16 >> %LOG% 2>&1
if errorlevel 1 (
  echo === SMOKE GATE FAILED, PC not started >> %LOG%
  exit /b 5
)
echo === PC %DATE% %TIME% >> %LOG%
%PY% -I cr1_trainer.py CR1_PC_P8 --mode p8 --updates %UPD% %BASE% >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\CR1_PC_P8\TRAIN_RECEIPT_V1.json" (
  echo === PC has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I cr1_round.py CR1_PC_P8 --mode pc >> %LOG% 2>&1
if errorlevel 1 (
  echo === ROUND PC rc=%ERRORLEVEL%: no frozen formula with corr ^> 0.7, PC NOT FOUND, chain stops >> %LOG%
  exit /b 20
)
echo === NULL %DATE% %TIME% >> %LOG%
%PY% -I cr1_trainer.py CR1_NULL --mode null --updates %UPD% %BASE% >> %LOG% 2>&1
echo === NULL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\CR1_NULL\TRAIN_RECEIPT_V1.json" (
  echo === NULL has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I cr1_round.py CR1_NULL --mode null --pc CR1_PC_P8 >> %LOG% 2>&1
if errorlevel 1 (
  echo === ROUND NULL rc=%ERRORLEVEL%: PC not above the NULL bar, PC NOT FOUND, chain stops >> %LOG%
  exit /b 21
)
echo === REAL %DATE% %TIME% >> %LOG%
%PY% -I cr1_trainer.py CR1_REAL --mode real --updates %UPD% %BASE% >> %LOG% 2>&1
echo === REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\CR1_REAL\TRAIN_RECEIPT_V1.json" (
  echo === REAL has no receipt, stops >> %LOG%
  exit /b 6
)
%PY% -I cr1_round.py CR1_REAL --mode real --null CR1_NULL >> %LOG% 2>&1
echo === ROUND REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
echo === CR1 END %DATE% %TIME% >> %LOG%
