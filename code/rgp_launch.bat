@echo off
rem TV1 RGP search layer, p8 (the realistic plant level the RL rounds could not reach).
rem Gradient-free: fitness is TRAIN gross bp/day over all 50 tiles, no policy gradient.
rem Refuses to start if the run dir already holds a freeze or a log.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN_DIR=%ROOT%\artifacts\m2h_v1\RGP_P8_V1
set LOG=D:\ChengboRemote\runtime\basin3_h1\RGP_P8_V1.log
if exist "%RUN_DIR%\FROZEN_RGP_TOP5.json" (
  echo === REFUSED %DATE% %TIME%: FROZEN_RGP_TOP5.json exists >> %LOG%
  exit /b 3
)
if exist "%RUN_DIR%\RGP_LOG.jsonl" (
  echo === REFUSED %DATE% %TIME%: RGP_LOG.jsonl exists >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I m2h_rgp.py --out RGP_P8_V1 --mode p8 --seed-run TV1_R2D_DENSE_P32 --rounds 10 --pop 512 --keep 96 --seed 20261011 --max-hours 12 >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
