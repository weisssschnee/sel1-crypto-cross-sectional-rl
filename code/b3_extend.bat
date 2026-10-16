@echo off
rem BASIN3_H1_20261009 extension: resume from checkpoint_0127 to 960 updates inside the 9 h budget.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN_DIR=D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\BASIN3_H1_20261009
set LOG=D:\ChengboRemote\runtime\basin3_h1\BASIN3_H1_20261009.log
if exist "%RUN_DIR%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: TRAIN_RECEIPT_V1.json exists >> %LOG%
  exit /b 3
)
echo === EXTEND %DATE% %TIME% >> %LOG%
%PY% -I m2h_trainer.py BASIN3_H1_20261009 --updates 960 --batch 256 --seed 20261009 --max-hours 9 --resume auto >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
