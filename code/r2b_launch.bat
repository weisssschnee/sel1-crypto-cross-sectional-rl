@echo off
rem TV1 R2b: dense v2 knobs (adv deadband 0.5 + entropy 0.15) on the p8 plant, as an A/B against
rem R2. Refuses if the run already has a receipt or a checkpoint.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN_DIR=%ROOT%\artifacts\m2h_v1\TV1_R2B_DENSE_P8
set LOG=D:\ChengboRemote\runtime\basin3_h1\TV1_R2B_DENSE_P8.log
if exist "%RUN_DIR%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: TRAIN_RECEIPT_V1.json exists >> %LOG%
  exit /b 3
)
if exist "%RUN_DIR%\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: checkpoints exist >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I m2h_plant.py --trainer dense2 --mode p8 -- TV1_R2B_DENSE_P8 --updates 128 --batch 128 --tiles 16 --seed 20261009 --max-hours 2.5 --reward ic --select full --ent-coef 0.15 --adv-deadband 0.5 >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
