@echo off
rem BASIN5 debug round (merged plan 10-13, user: all approved). Smoke p32 16 updates
rem (B5_DBG_SMOKE_P32) -> b5_smoke_gate + b5_dbg_gate -> debug positive control p32 256 updates
rem (B5_DBG_PC_P32) -> b5_dbg_after.bat (waits for B5_GATE1\DECISION.json, HoF round report,
rem found -> real-data formal round). Refuses if either dir has a receipt or a checkpoint, or the
rem pre-registration RUN_CONFIG.json is missing. Resume: b5_dbg_pc_resume.bat.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_DBG.log
set SM=B5_DBG_SMOKE_P32
set PC=B5_DBG_PC_P32
set BASE=--batch 128 --tiles 16 --seed 20261009 --reward gross_t --select full
if exist "%A%\%SM%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: %SM% receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\%SM%\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: %SM% checkpoints exist >> %LOG%
  exit /b 3
)
if exist "%A%\%PC%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: %PC% receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\%PC%\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: %PC% checkpoints exist >> %LOG%
  exit /b 3
)
if not exist "%A%\%PC%\RUN_CONFIG.json" (
  echo === REFUSED %DATE% %TIME%: %PC%\RUN_CONFIG.json missing >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% > %LOG%
echo === SMOKE %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant_dbg.py --mode p32 -- %SM% --updates 16 --max-hours 1.5 %BASE% >> %LOG% 2>&1
echo === SMOKE rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I b5_smoke_gate.py %SM% --updates 16 >> %LOG% 2>&1
if errorlevel 1 (
  echo === SMOKE GATE FAILED, PC not started >> %LOG%
  exit /b 5
)
%PY% -I b5_dbg_gate.py %SM% >> %LOG% 2>&1
if errorlevel 1 (
  echo === DBG GATE FAILED, PC not started >> %LOG%
  exit /b 5
)
echo === PC %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant_dbg.py --mode p32 -- %PC% --updates 256 --max-hours 10 %BASE% >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\%PC%\TRAIN_RECEIPT_V1.json" (
  echo === PC has no receipt, stops ^(resume: b5_dbg_pc_resume.bat^) >> %LOG%
  exit /b 6
)
call D:\ChengboRemote\runtime\basin3_h1\b5_dbg_after.bat
echo === END %DATE% %TIME% >> %LOG%
