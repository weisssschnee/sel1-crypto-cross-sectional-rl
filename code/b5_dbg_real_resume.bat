@echo off
rem BASIN5 formal real-data round resume (--resume auto), then b5_dbg_real_post.bat.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_DBG.log
set REAL=B5_DBG_REAL
if exist "%A%\%REAL%\TRAIN_RECEIPT_V1.json" (
  echo === REAL RESUME REFUSED %DATE% %TIME%: receipt exists >> %LOG%
  exit /b 3
)
if not exist "%A%\%REAL%\checkpoint_*.pt" (
  echo === REAL RESUME REFUSED %DATE% %TIME%: no checkpoint >> %LOG%
  exit /b 3
)
echo === REAL RESUME %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_trainer_dbg.py %REAL% --updates 960 --batch 128 --tiles 16 --seed 20261010 --max-hours 24 --reward gross_t --select full --resume auto >> %LOG% 2>&1
echo === REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
call D:\ChengboRemote\runtime\basin3_h1\b5_dbg_real_post.bat
