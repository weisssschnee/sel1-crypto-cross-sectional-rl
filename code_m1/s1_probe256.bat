@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
%PY% -u m2_trainer.py S1_PROBE256_20261008 --shard-dir D:\ChengboRemote\data\m1_smoke_2025-07 --updates 1 --batch 256 --windows 4 --elite-recheck 0 --cost flat6 > D:\ChengboRemote\runtime\m1_stage1\probe256.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\probe256.log
