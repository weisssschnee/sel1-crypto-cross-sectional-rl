@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set S1_RUN=S1_RESUME2_A
set S1_FINAL=D:\ChengboRemote\runtime\m1_stage1\fb_smoke\FINAL_BATCH.jsonl
set S1_LIMIT=8
set S1_OUT=D:/ChengboRemote/runtime/m1_stage1/paired_smoke.json
%PY% -u s1_paired_eval.py > D:\ChengboRemote\runtime\m1_stage1\paired_smoke.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\paired_smoke.log
