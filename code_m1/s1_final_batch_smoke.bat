@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set S1_RUN=S1_RESUME2_A
set S1_N=64
set S1_OUT_DIR=D:\ChengboRemote\runtime\m1_stage1\fb_smoke
%PY% -m pyflakes s1_final_batch.py > D:\ChengboRemote\runtime\m1_stage1\fb_smoke.log 2>&1
%PY% -u s1_final_batch.py >> D:\ChengboRemote\runtime\m1_stage1\fb_smoke.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\fb_smoke.log
