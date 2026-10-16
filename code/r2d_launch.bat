@echo off
rem TV1 R2d: dense v1, IC reward, p32 STRONG plant. Fills the missing {dense x p32} cell:
rem old/p32=R1 found, old/p8=R3, dense/p8=R2 not found. Refuses if receipt or checkpoints exist.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN_DIR=%ROOT%\artifacts\m2h_v1\TV1_R2D_DENSE_P32
set LOG=D:\ChengboRemote\runtime\basin3_h1\TV1_R2D_DENSE_P32.log
if exist "%RUN_DIR%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: receipt exists >> %LOG%
  exit /b 3
)
if exist "%RUN_DIR%\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: checkpoints exist >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I m2h_plant.py --trainer dense --mode p32 -- TV1_R2D_DENSE_P32 --updates 128 --batch 128 --tiles 16 --seed 20261009 --max-hours 2.5 --reward ic --select full >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
