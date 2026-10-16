@echo off
rem BASIN5_PC_ENT positive control resume: b5_pc_ent.bat's PC arguments plus --resume auto
rem (alpha comes back from the checkpoint). Refuses if the receipt exists or there is no checkpoint.
rem Then b5_ent_after.bat if present.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_PC_ENT.log
set PC=B5_PC_P8_ENT
set BASE=--batch 128 --tiles 16 --seed 20261009 --reward gross_t --select full --ent-eta 0.05 --ent-alpha-max 1.0
if exist "%A%\%PC%\TRAIN_RECEIPT_V1.json" (
  echo === RESUME REFUSED %DATE% %TIME%: %PC% receipt exists >> %LOG%
  exit /b 3
)
if not exist "%A%\%PC%\checkpoint_*.pt" (
  echo === RESUME REFUSED %DATE% %TIME%: no %PC% checkpoint >> %LOG%
  exit /b 3
)
echo === PC RESUME %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_plant_ent.py --mode p8 -- %PC% --updates 256 --max-hours 10 --ent-target 2.0 %BASE% --resume auto >> %LOG% 2>&1
echo === PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\%PC%\TRAIN_RECEIPT_V1.json" exit /b 6
if exist D:\ChengboRemote\runtime\basin3_h1\b5_ent_after.bat (
  echo === AFTER %DATE% %TIME% >> %LOG%
  call D:\ChengboRemote\runtime\basin3_h1\b5_ent_after.bat
)
echo === END %DATE% %TIME% >> %LOG%
