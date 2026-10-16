@echo off
rem TV1 R2c: dense v2, GROSS reward (cost0), p8 plant. Fills the reward dimension after R2/R2b
rem (IC) failed. Refuses if the run already has a receipt or a checkpoint.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN_DIR=%ROOT%\artifacts\m2h_v1\TV1_R2C_DENSE_P8_GROSS
set LOG=D:\ChengboRemote\runtime\basin3_h1\TV1_R2C_DENSE_P8_GROSS.log
if exist "%RUN_DIR%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: receipt exists >> %LOG%
  exit /b 3
)
if exist "%RUN_DIR%\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: checkpoints exist >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I m2h_plant.py --trainer dense2 --mode p8 -- TV1_R2C_DENSE_P8_GROSS --updates 128 --batch 128 --tiles 16 --seed 20261009 --max-hours 2.5 --reward gross --select full --ent-coef 0.15 --adv-deadband 0.5 >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
