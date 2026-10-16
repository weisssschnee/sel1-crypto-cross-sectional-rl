@echo off
rem TV1 night chain: R2d resume to 256 -> R2d report -> R5 (dense, 50 fixed tiles, p8) -> R5 report.
rem R2d resume goes through m2h_plant (the planted shard is rebuilt each process; resuming the
rem bare trainer would train on unplanted data). R5: --tiles 50 = all 50 TRAIN tiles every update
rem (choice(50, size=50, replace=False) sorted), no code change.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\r2d_r5_chain.log
echo === START %DATE% %TIME% >> %LOG%

rem --- R2d resume: needs checkpoints, and archives the u128 receipt/report first
if not exist "%A%\TV1_R2D_DENSE_P32\checkpoint_0127.pt" (
  echo === REFUSED %DATE% %TIME%: no R2d checkpoint_0127 >> %LOG%
  exit /b 3
)
if exist "%A%\TV1_R2D_DENSE_P32\TRAIN_RECEIPT_V1.json" (
  copy /y "%A%\TV1_R2D_DENSE_P32\TRAIN_RECEIPT_V1.json" "%A%\TV1_R2D_DENSE_P32\TRAIN_RECEIPT_u128.json" >> %LOG% 2>&1
  del "%A%\TV1_R2D_DENSE_P32\TRAIN_RECEIPT_V1.json"
)
if exist "%A%\TV1_R2D_DENSE_P32\ROUND_REPORT.json" (
  copy /y "%A%\TV1_R2D_DENSE_P32\ROUND_REPORT.json" "%A%\TV1_R2D_DENSE_P32\ROUND_REPORT_u127.json" >> %LOG% 2>&1
  del "%A%\TV1_R2D_DENSE_P32\ROUND_REPORT.json"
)
echo === R2D RESUME %DATE% %TIME% >> %LOG%
%PY% -I m2h_plant.py --trainer dense --mode p32 -- TV1_R2D_DENSE_P32 --updates 256 --batch 128 --tiles 16 --seed 20261009 --max-hours 3 --reward ic --select full --resume auto >> %LOG% 2>&1
echo === R2D rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist "%A%\TV1_R2D_DENSE_P32\TRAIN_RECEIPT_V1.json" (
  if not exist "%A%\TV1_R2D_DENSE_P32\STOPPED.json" (
    echo === R2D FAILED, chain stops >> %LOG%
    exit /b 4
  )
)
%PY% -I tv1_round.py TV1_R2D_DENSE_P32 --trainer dense --mode p32 >> %LOG% 2>&1
echo === R2D REPORT rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%

rem --- R5: dense v1, all 50 tiles, p8. Refuses on existing artifacts.
if exist "%A%\TV1_R5_DENSE_P8_T50\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: R5 receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\TV1_R5_DENSE_P8_T50\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: R5 checkpoints exist >> %LOG%
  exit /b 3
)
echo === R5 START %DATE% %TIME% >> %LOG%
%PY% -I m2h_plant.py --trainer dense --mode p8 -- TV1_R5_DENSE_P8_T50 --updates 256 --batch 128 --tiles 50 --seed 20261009 --max-hours 12 --reward ic --select full >> %LOG% 2>&1
echo === R5 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I tv1_round.py TV1_R5_DENSE_P8_T50 --trainer dense --mode p8 >> %LOG% 2>&1
echo === R5 REPORT rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
echo === CHAIN END %DATE% %TIME% >> %LOG%
