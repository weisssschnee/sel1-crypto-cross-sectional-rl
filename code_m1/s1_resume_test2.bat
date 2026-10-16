@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set LOG=D:\ChengboRemote\runtime\m1_stage1\resume_test2.log
set COMMON=--shard-dir D:\ChengboRemote\data\m1_smoke_2025-07 --batch 64 --windows 4 --elite-recheck 4 --cost flat6 --checkpoint-every 2 --seed 777
echo === A: uninterrupted 4 updates > %LOG%
%PY% -u m2_trainer.py S1_RESUME2_A --updates 4 %COMMON% >> %LOG% 2>&1
echo EXIT_A=%ERRORLEVEL% >> %LOG%
echo === B1: max-hours ceiling stops after the first update >> %LOG%
%PY% -u m2_trainer.py S1_RESUME2_B --updates 4 --max-hours 0.001 %COMMON% >> %LOG% 2>&1
echo EXIT_B1=%ERRORLEVEL% >> %LOG%
echo === B2: --resume auto finishes the run >> %LOG%
%PY% -u m2_trainer.py S1_RESUME2_B --updates 4 --resume auto %COMMON% >> %LOG% 2>&1
echo EXIT_B2=%ERRORLEVEL% >> %LOG%
echo === B3: resume must REFUSE a changed argument (seed) >> %LOG%
%PY% -u m2_trainer.py S1_RESUME2_B --updates 4 --resume auto --shard-dir D:\ChengboRemote\data\m1_smoke_2025-07 --batch 64 --windows 4 --elite-recheck 4 --cost flat6 --checkpoint-every 2 --seed 778 >> %LOG% 2>&1
echo EXIT_B3=%ERRORLEVEL% >> %LOG%
echo === DRYRUN: the real launch command, manifest only >> %LOG%
%PY% -u m2_trainer.py S1_STAGE1_DRYRUN_20261008 --shard-dir D:\ChengboRemote\data\m1_shards_v2 --updates 128 --batch 256 --windows 4 --elite-recheck 32 --cost flat6 --aggregate mean --max-hours 9 --manifest-only >> %LOG% 2>&1
echo EXIT_DRY=%ERRORLEVEL% >> %LOG%
echo ALL_DONE >> %LOG%
