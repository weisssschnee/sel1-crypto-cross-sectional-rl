@echo off
rem BASIN5 s5: smoke (p8, 16 updates) -> b5_smoke_gate.py -> positive control (p8, 256 updates)
rem -> b5_after_pc.bat if it exists by then (round report / next stage, written while PC runs).
rem Reward b (gross_t), select full, ent 0.03 / no deadband (the dense trainer's own values).
rem Refuses if either run dir already has a receipt or a checkpoint. PC resume: b5_pc_resume.bat.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_CHAIN.log
if exist "%A%\B5_SMOKE_P8\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: B5_SMOKE_P8 receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\B5_SMOKE_P8\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: B5_SMOKE_P8 checkpoints exist >> %LOG%
  exit /b 3
)
if exist "%A%\B5_PC_P8\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: B5_PC_P8 receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\B5_PC_P8\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: B5_PC_P8 checkpoints exist >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_grammar.py D:\ChengboRemote\runtime\basin3_h1\B5_GRAMMAR_SELFTEST.json >> %LOG% 2>&1
if errorlevel 1 (
  echo === GRAMMAR SELFTEST FAILED, chain stops >> %LOG%
  exit /b 4
)
echo === SMOKE %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant.py --mode p8 -- B5_SMOKE_P8 --updates 16 --batch 128 --tiles 16 --seed 20261009 --max-hours 1.5 --reward gross_t --select full >> %LOG% 2>&1
echo === SMOKE rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I b5_smoke_gate.py B5_SMOKE_P8 --updates 16 >> %LOG% 2>&1
if errorlevel 1 (
  echo === SMOKE GATE FAILED, chain stops >> %LOG%
  exit /b 5
)
echo === PC %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant.py --mode p8 -- B5_PC_P8 --updates 256 --batch 128 --tiles 16 --seed 20261009 --max-hours 10 --reward gross_t --select full >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\B5_PC_P8\TRAIN_RECEIPT_V1.json" (
  echo === PC has no receipt, chain stops ^(resume: b5_pc_resume.bat^) >> %LOG%
  exit /b 6
)
if exist D:\ChengboRemote\runtime\basin3_h1\b5_after_pc.bat (
  echo === AFTER_PC %DATE% %TIME% >> %LOG%
  call D:\ChengboRemote\runtime\basin3_h1\b5_after_pc.bat
)
echo === END %DATE% %TIME% >> %LOG%
