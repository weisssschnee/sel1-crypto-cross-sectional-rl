@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
%PY% -u m2_trainer.py S1_PROBE256_B25_20261008 --shard-dir D:\ChengboRemote\data\m1_smoke_2025-07 --updates 1 --batch 256 --windows 4 --elite-recheck 0 --cost flat6 --vm-block-elems 33554432 > D:\ChengboRemote\runtime\m1_stage1\probe256_b25.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\probe256_b25.log
%PY% -u m2_trainer.py S1_PROBE256_B23_20261008 --shard-dir D:\ChengboRemote\data\m1_smoke_2025-07 --updates 1 --batch 256 --windows 4 --elite-recheck 0 --cost flat6 --vm-block-elems 8388608 > D:\ChengboRemote\runtime\m1_stage1\probe256_b23.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\probe256_b23.log
