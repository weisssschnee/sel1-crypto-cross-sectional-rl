@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set S1_RUN=S1_PROBE256_FINAL_20261008
set S1_FINAL=D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2_v1\S1_PROBE256_FINAL_20261008\FORMULAS.jsonl
set S1_OUT=D:/ChengboRemote/runtime/m1_stage1/paired_timing.json
%PY% -u s1_paired_eval.py > D:\ChengboRemote\runtime\m1_stage1\paired_timing.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\paired_timing.log
