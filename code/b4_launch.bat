@echo off
rem BASIN4 launch (TV1_A2_BASIN4_ORDER s3), real TRAIN data, no plant. %1 = run name.
rem Refuses if the run already has a receipt or a checkpoint; resume is b4_resume.bat.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN=%1
set RUN_DIR=%ROOT%\artifacts\m2h_v1\%RUN%
set LOG=D:\ChengboRemote\runtime\basin3_h1\%RUN%.log
if "%RUN%"=="" exit /b 2
if exist "%RUN_DIR%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: TRAIN_RECEIPT_V1.json exists >> %LOG%
  exit /b 3
)
if exist "%RUN_DIR%\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: checkpoints exist, use b4_resume.bat >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I m2h_dense_trainer.py %RUN% --updates 960 --batch 128 --tiles 16 --seed 20261010 --max-hours 24 --reward ic --select full >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
