@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set S1_RUN=S1_RESUME_A
set S1_TOPK=6
set S1_MONTHS=2026-02,2026-03
set S1_OUT=D:/ChengboRemote/runtime/m1_stage1/s1_dev_eval_smoke.json
set LOG=D:\ChengboRemote\runtime\m1_stage1\dev_eval_smoke.log
%PY% -u s1_dev_eval.py > %LOG% 2>&1
echo EXIT=%ERRORLEVEL% >> %LOG%
