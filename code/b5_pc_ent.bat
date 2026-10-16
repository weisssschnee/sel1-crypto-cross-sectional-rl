@echo off
rem BASIN5_PC_ENT (BASIN5_PC_ENT_ORDER): smoke p8 16 updates in its own dir (H* 7.0 = controller
rem test only: every family starts below 7 so alpha has to move) -> b5_smoke_gate + b5_ent_gate ->
rem positive control p8 256 updates, H* 2.0, eta 0.05, alpha in [0.03, 1.0] -> b5_ent_after.bat if
rem present (round report). Every other argument = B5_PC_P8. Refuses if either run dir has a receipt
rem or a checkpoint. Resume: b5_pc_ent_resume.bat. No real round, no null, no DEV after it.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_PC_ENT.log
set SM=B5_SMOKE_P8_ENT
set PC=B5_PC_P8_ENT
set BASE=--batch 128 --tiles 16 --seed 20261009 --reward gross_t --select full --ent-eta 0.05 --ent-alpha-max 1.0
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
  echo === REFUSED %DATE% %TIME%: %PC%\RUN_CONFIG.json ^(pre-registration^) missing >> %LOG%
  exit /b 3
)
rem log "w": the START line truncates it; REFUSED lines above only append
echo === START %DATE% %TIME% > %LOG%
echo === SMOKE %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant_ent.py --mode p8 -- %SM% --updates 16 --max-hours 1.5 --ent-target 7.0 %BASE% >> %LOG% 2>&1
echo === SMOKE rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I b5_smoke_gate.py %SM% --updates 16 >> %LOG% 2>&1
if errorlevel 1 (
  echo === SMOKE GATE FAILED, PC not started >> %LOG%
  exit /b 5
)
%PY% -I b5_ent_gate.py %SM% >> %LOG% 2>&1
if errorlevel 1 (
  echo === ENT GATE FAILED, PC not started >> %LOG%
  exit /b 5
)
echo === PC %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant_ent.py --mode p8 -- %PC% --updates 256 --max-hours 10 --ent-target 2.0 %BASE% >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\%PC%\TRAIN_RECEIPT_V1.json" (
  echo === PC has no receipt, stops ^(resume: b5_pc_ent_resume.bat^) >> %LOG%
  exit /b 6
)
if exist D:\ChengboRemote\runtime\basin3_h1\b5_ent_after.bat (
  echo === AFTER %DATE% %TIME% >> %LOG%
  call D:\ChengboRemote\runtime\basin3_h1\b5_ent_after.bat
)
echo === END %DATE% %TIME% >> %LOG%
