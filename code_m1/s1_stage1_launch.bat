@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set RUN_DIR=D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2_v1\S1_STAGE1_20261008
set LOG=D:\ChengboRemote\runtime\m1_stage1\S1_STAGE1_20261008.log
if exist "%RUN_DIR%\TRAIN_RECEIPT_V1.json" (
  echo === REFUSED %DATE% %TIME%: TRAIN_RECEIPT_V1.json exists >> %LOG%
  exit /b 3
)
if exist "%RUN_DIR%\checkpoint_*.pt" (
  echo === REFUSED %DATE% %TIME%: checkpoints exist, resume needs its own bat with --resume auto >> %LOG%
  exit /b 4
)
echo === LAUNCH %DATE% %TIME% >> %LOG%
%PY% -u m2_trainer.py S1_STAGE1_20261008 --shard-dir D:\ChengboRemote\data\m1_shards_v2 --updates 128 --batch 256 --windows 4 --elite-recheck 32 --cost flat6 --aggregate mean --max-hours 9 >> %LOG% 2>&1
echo EXIT=%ERRORLEVEL% >> %LOG%
