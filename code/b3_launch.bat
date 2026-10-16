@echo off
rem BASIN3_H1_20261009 launch. Refuses to start if the run already produced a receipt or a
rem checkpoint, so a duplicate scheduled start cannot overwrite a finished or in-progress run.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN_DIR=D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\BASIN3_H1_20261009
set LOG=D:\ChengboRemote\runtime\basin3_h1\BASIN3_H1_20261009.log
if exist "%RUN_DIR%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: TRAIN_RECEIPT_V1.json exists >> %LOG%
  exit /b 3
)
if exist "%RUN_DIR%\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: checkpoints exist, use b3_resume.bat >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I m2h_trainer.py BASIN3_H1_20261009 --updates 128 --batch 256 --seed 20261009 --max-hours 9 >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
