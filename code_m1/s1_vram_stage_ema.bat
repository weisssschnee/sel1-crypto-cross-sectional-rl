@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set S1_EMA_MAX=1
set S1_TAG=_EMA1
%PY% -u s1_vram_stage.py > D:\ChengboRemote\runtime\m1_stage1\vram_stage_ema1.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\vram_stage_ema1.log
