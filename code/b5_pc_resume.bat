@echo off
rem BASIN5 positive control resume: same args as b5_chain.bat's PC step plus --resume auto.
rem Refuses if the receipt exists or there is no checkpoint. Then b5_after_pc.bat if present.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_CHAIN.log
if exist "%A%\B5_PC_P8\TRAIN_RECEIPT_V1.json" (
  echo === RESUME REFUSED %DATE% %TIME%: B5_PC_P8 receipt exists >> %LOG%
  exit /b 3
)
if not exist "%A%\B5_PC_P8\checkpoint_*.pt" (
  echo === RESUME REFUSED %DATE% %TIME%: no B5_PC_P8 checkpoint >> %LOG%
  exit /b 3
)
echo === PC RESUME %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant.py --mode p8 -- B5_PC_P8 --updates 256 --batch 128 --tiles 16 --seed 20261009 --max-hours 10 --reward gross_t --select full --resume auto >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\B5_PC_P8\TRAIN_RECEIPT_V1.json" exit /b 6
if exist D:\ChengboRemote\runtime\basin3_h1\b5_after_pc.bat (
  echo === AFTER_PC %DATE% %TIME% >> %LOG%
  call D:\ChengboRemote\runtime\basin3_h1\b5_after_pc.bat
)
echo === END %DATE% %TIME% >> %LOG%
