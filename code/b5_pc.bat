@echo off
rem BASIN5 s5 positive control after the smoke (b5_chain.bat stopped on the first gate version,
rem T1 mean n_valid 1.9 < 5; gate revised to "valid in the last 4 updates", see SMOKE_GATE.json).
rem Re-runs b5_smoke_gate.py on B5_SMOKE_P8, then p8 256 updates, then b5_after_pc.bat if present.
rem Refuses if B5_PC_P8 already has a receipt or a checkpoint. Resume: b5_pc_resume.bat.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_CHAIN.log
if exist "%A%\B5_PC_P8\TRAIN_RECEIPT_V1.json" (
  echo === PC REFUSED %DATE% %TIME%: B5_PC_P8 receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\B5_PC_P8\checkpoint_*.pt" (
  echo === PC REFUSED %DATE% %TIME%: B5_PC_P8 checkpoints exist >> %LOG%
  exit /b 3
)
echo === GATE2 %DATE% %TIME% >> %LOG%
%PY% -I b5_smoke_gate.py B5_SMOKE_P8 --updates 16 >> %LOG% 2>&1
if errorlevel 1 (
  echo === SMOKE GATE v2 FAILED, PC not started >> %LOG%
  exit /b 5
)
echo === PC %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant.py --mode p8 -- B5_PC_P8 --updates 256 --batch 128 --tiles 16 --seed 20261009 --max-hours 10 --reward gross_t --select full >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\B5_PC_P8\TRAIN_RECEIPT_V1.json" (
  echo === PC has no receipt, stops ^(resume: b5_pc_resume.bat^) >> %LOG%
  exit /b 6
)
if exist D:\ChengboRemote\runtime\basin3_h1\b5_after_pc.bat (
  echo === AFTER_PC %DATE% %TIME% >> %LOG%
  call D:\ChengboRemote\runtime\basin3_h1\b5_after_pc.bat
)
echo === END %DATE% %TIME% >> %LOG%
