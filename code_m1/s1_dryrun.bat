@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
%PY% m2_trainer.py S1_MANIFEST_DRYRUN_20261008 --shard-dir D:\ChengboRemote\data\m1_shards_v2 --cost depth --depth-dir D:\ChengboRemote\runtime\p1 --manifest-only
