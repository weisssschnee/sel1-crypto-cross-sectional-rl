@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
%PY% -u m2_trainer.py S1_TRAIN_SMOKE_20261008 --shard-dir D:\ChengboRemote\data\m1_smoke_2025-07 --updates 2 --batch 32 --windows 4 --elite-recheck 4 --cost depth --depth-dir D:\ChengboRemote\runtime\p1 > D:\ChengboRemote\runtime\m1_stage1\train_smoke.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\train_smoke.log
