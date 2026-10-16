@echo off
rem BASIN4 resume: --resume auto, newest checkpoint; refuses if the run is finished.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN=%1
set RUN_DIR=%ROOT%\artifacts\m2h_v1\%RUN%
set LOG=D:\ChengboRemote\runtime\basin3_h1\%RUN%.log
if "%RUN%"=="" exit /b 2
if exist "%RUN_DIR%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: TRAIN_RECEIPT_V1.json exists, run is finished >> %LOG%
  exit /b 3
)
echo === RESUME %DATE% %TIME% >> %LOG%
%PY% -I m2h_dense_trainer.py %RUN% --updates 960 --batch 128 --tiles 16 --seed 20261010 --max-hours 24 --reward ic --select full --resume auto >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
