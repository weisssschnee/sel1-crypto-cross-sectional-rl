@echo off
rem BASIN5 debug PC resume: b5_dbg_chain.bat's PC arguments plus --resume auto, then b5_dbg_after.bat.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_DBG.log
set PC=B5_DBG_PC_P32
set BASE=--batch 128 --tiles 16 --seed 20261009 --reward gross_t --select full
if exist "%A%\%PC%\TRAIN_RECEIPT_V1.json" (
  echo === RESUME REFUSED %DATE% %TIME%: receipt exists >> %LOG%
  exit /b 3
)
if not exist "%A%\%PC%\checkpoint_*.pt" (
  echo === RESUME REFUSED %DATE% %TIME%: no checkpoint >> %LOG%
  exit /b 3
)
echo === PC RESUME %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant_dbg.py --mode p32 -- %PC% --updates 256 --max-hours 10 %BASE% --resume auto >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\%PC%\TRAIN_RECEIPT_V1.json" exit /b 6
call D:\ChengboRemote\runtime\basin3_h1\b5_dbg_after.bat
